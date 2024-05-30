import torch
import torch.nn as nn
from transformers import DebertaV2Tokenizer, DebertaV2Model
from torch_geometric.nn.conv import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.data import Batch, Data
from torch_geometric.nn.norm import BatchNorm
import torch.nn.functional as func
from torch_geometric.nn.pool import global_mean_pool


#  the ZLPR loss for multi-label classification
#  https://arxiv.org/abs/2208.02955
def multilabel_categorical_cross_entropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    loss = neg_loss + pos_loss
    return loss.mean(0)


# the GNN encoder to encode graph information
class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_unit, gnn, dropout=0.0, batch_norm=False):
        super().__init__()

        self.num_unit = num_unit
        self.dropout = dropout
        self.batch_norm = batch_norm
        assert gnn in ['GCN', 'SAGE', 'GAT', 'GIN']
        gnn_map = {
            'GCN': GCNConv,
            'SAGE': SAGEConv,
            'GAT': GATConv,
            'GIN': GINConv
        }
        Conv = gnn_map[gnn]
        if gnn != 'GIN':
            in_conv = Conv(in_dim, hidden_dim)
        else:
            in_conv = Conv(nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                         nn.ReLU(),
                                         nn.Linear(hidden_dim, hidden_dim)))
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.convs.append(in_conv)

        for i in range(num_unit):
            if gnn != 'GIN':
                conv = Conv(hidden_dim, hidden_dim)
            else:
                conv = Conv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim)))
            bn = BatchNorm(hidden_dim)
            self.convs.append(conv)
            self.batch_norms.append(bn)

        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.activation(x)
            if self.batch_norm:
                x = self.batch_norms[i](x)
            x = func.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# the main module of each expert
class MyModel(nn.Module):
    def __init__(self, device, num_class, task, lm_path='microsoft/deberta-v3-large', max_length=3840,
                 text_augmentation=None, graph_augmentation=None):
        super().__init__()

        # we employ deberta-v3 as the encode lm
        # from https://huggingface.co/microsoft/deberta-v3-large
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(lm_path)
        self.lm = DebertaV2Model.from_pretrained(lm_path)

        # we froze the model parameters of the encode lm
        for name, param in self.lm.named_parameters():
            param.requires_grad = False
        self.max_length = max_length

        self.graph_encoder = GNNEncoder(1024, 1024, 2, gnn='GIN', dropout=0.5)

        # specify the field of expert
        self.text_augmentation = text_augmentation
        self.graph_augmentation = graph_augmentation

        if text_augmentation is None:
            self.text_transfer = nn.Linear(1024, 1024)
        else:
            self.text_transfer = nn.Linear(1024 * 2, 1024)
        if graph_augmentation is None:
            self.graph_transfer = nn.Linear(1024, 1024)
        else:
            self.graph_transfer = nn.Linear(1024 * 2, 1024)

        self.cls = nn.Sequential(
            nn.Linear(1024 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_class)
        )
        self.device = device
        self.pooling = global_mean_pool

        # cross entropy loss for binary classification
        if task == 'TASK1':
            self.loss_fn = nn.CrossEntropyLoss()
        # ZLPR loss for multi-label classification
        else:
            self.loss_fn = multilabel_categorical_cross_entropy

    # tokenize text into input_ids
    def tokenize(self, data):
        tokens = []
        for item in data:
            token = self.tokenizer.tokenize(item)
            if len(token) == 0:
                token = [self.tokenizer.pad_token_id]
            token = token[:self.max_length-2]
            tokens.append([self.tokenizer.cls_token_id] +
                          self.tokenizer.encode(token, add_special_tokens=False) +
                          [self.tokenizer.eos_token_id])
        max_length = 0
        for token in tokens:
            max_length = max(max_length, len(token))
        input_ids = []
        token_type_ids = []
        attention_mask = []
        for token in tokens:
            input_ids.append(token + [self.tokenizer.pad_token_id] * (max_length - len(token)))
            token_type_ids.append([0] * max_length)
            attention_mask.append([1] * len(token) + [0] * (max_length - len(token)))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return {
            'input_ids': input_ids.to(self.device),
            'token_type_ids': token_type_ids.to(self.device),
            'attention_mask': attention_mask.to(self.device)
        }

    def forward(self, data):
        text = data['text']
        text_input = self.tokenize(text)
        text_reps = self.lm(**text_input).last_hidden_state.mean(dim=1)
        if self.text_augmentation is not None:
            text_augmentation = data[self.text_augmentation]
            text_augmentation_input = self.tokenize(text_augmentation)
            text_augmentation_reps = self.lm(**text_augmentation_input).last_hidden_state.mean(dim=1)
            text_reps = self.text_transfer(
                torch.cat([text_reps, text_augmentation_reps], dim=-1)
            )
        else:
            text_reps = self.text_transfer(text_reps)
        graphs = []
        for _, graph_info in enumerate(data['graph_info']):
            comment = graph_info['comment']
            comment_input = self.tokenize(comment)
            comment_reps = self.lm(**comment_input).last_hidden_state.mean(dim=1)
            if self.graph_augmentation is not None:
                graph_augmentation = graph_info[self.graph_augmentation]
                graph_augmentation_input = self.tokenize(graph_augmentation)
                graph_augmentation_reps = self.lm(**graph_augmentation_input).last_hidden_state.mean(dim=1)
                comment_reps = self.graph_transfer(
                    torch.cat([comment_reps, graph_augmentation_reps], dim=-1)
                )
            else:
                comment_reps = self.graph_transfer(comment_reps)
            x = torch.cat([text_reps[_].unsqueeze(0), comment_reps], dim=0)
            edge_index = graph_info['edge_index']
            graphs.append(Data(x=x, edge_index=edge_index).to(self.device))
        graph = Batch.from_data_list(graphs)
        reps = self.graph_encoder(graph.x, graph.edge_index)
        reps = self.pooling(reps, graph.batch)
        reps = torch.cat([text_reps, reps], dim=-1)

        pred = self.cls(reps)

        loss = self.loss_fn(pred, data['label'].to(self.device))

        return pred, loss
