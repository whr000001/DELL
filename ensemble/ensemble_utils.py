# we provide three non LLMs-based ensemble methods, namely, majority vote, likelihood weighted vote,
# and train on validation set.
import random
import torch
import torch.nn as nn
from einops import repeat
from sklearn.metrics import f1_score


def get_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro') * 100


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


class MyModel(nn.Module):
    def __init__(self, k, task):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(k))
        if task == 'TASK1':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = multilabel_categorical_cross_entropy

    def forward(self, x, y):
        batch_size = len(x)
        weight = torch.softmax(self.weight, dim=0)
        weight = repeat(weight, 'n -> b n', b=batch_size)
        x = torch.einsum('bi, bij->bj', weight, x)
        pred = x

        loss = self.loss_fn(pred, y)
        return pred, loss


def majority_vote(task, dataset):
    text_augmentation = ['None', 'emotion', 'framing', 'propaganda', 'retrieval', 'None', 'None']
    graph_augmentation = ['None', 'None', 'None', 'None', 'None', 'stance', 'relation']
    preds = []
    all_data = []
    for text, graph in zip(text_augmentation, graph_augmentation):
        data = torch.load(f'../expert/results/{task}_{dataset}_{text}_{graph}_test.pt')
        all_data.append(data)
        preds.append(data[1])
    preds = torch.stack(preds).to(torch.float).mean(0)
    preds = torch.greater(preds, 0.5).to(torch.long)
    preds = preds.numpy()
    return preds


def likelihood_weight(task, dataset):
    text_augmentation = ['None', 'emotion', 'framing', 'propaganda', 'retrieval', 'None', 'None']
    graph_augmentation = ['None', 'None', 'None', 'None', 'None', 'stance', 'relation']
    preds = []
    weights = []
    all_data = []
    for text, graph in zip(text_augmentation, graph_augmentation):
        data = torch.load(f'../expert/results/{task}_{dataset}_{text}_{graph}_test.pt')
        all_data.append(data)
        if task == 'TASK1':
            likelihood = data[0]
            likelihood = torch.softmax(likelihood, dim=-1)
            likelihood = torch.max(likelihood, dim=-1)[0]
        else:
            likelihood = data[0]
            likelihood = torch.softmax(likelihood, dim=-1)
        preds.append(data[1])
        weights.append(likelihood)
    preds = torch.stack(preds)
    weights = torch.stack(weights)
    pred_weight = preds * weights
    preds = pred_weight.sum(0) / weights.sum(0)
    preds = torch.greater(preds, 0.5).to(torch.long)
    preds = preds.numpy()
    return preds


def train(model, train_x, train_y, val_x, val_y, task):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_acc = 0
    best_state = model.state_dict()
    for i in range(5000):
        model.train()
        optimizer.zero_grad()
        _, loss = model(train_x, train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        out, _ = model(val_x, val_y)
        if task == 'TASK1':
            preds = out.argmax(-1).to('cpu').numpy()
        else:
            preds = (out > 0).to(torch.long).to('cpu').numpy()
        label = val_y.to('cpu').numpy()
        acc = get_metric(label, preds)
        if acc > best_acc:
            best_acc = acc
            for key, value in model.state_dict().items():
                best_state[key] = value.clone()
    model.load_state_dict(best_state)


def train_on_validation(task, dataset):
    device = torch.device('cpu')
    text_augmentation = ['None', 'emotion', 'framing', 'propaganda', 'retrieval', 'None', 'None']
    graph_augmentation = ['None', 'None', 'None', 'None', 'None', 'stance', 'relation']
    all_data = []
    all_data_val = []
    val_x = []
    test_x = []
    for text, graph in zip(text_augmentation, graph_augmentation):
        data = torch.load(f'../expert/results/{task}_{dataset}_{text}_{graph}_val.pt')
        if task == 'TASK1':
            val_x.append(data[0])
        else:
            val_x.append(data[1])
        all_data_val.append(data)
        data = torch.load(f'../expert/results/{task}_{dataset}_{text}_{graph}_test.pt')
        if task == 'TASK1':
            test_x.append(data[0])
        else:
            test_x.append(data[1])
        all_data.append(data)
    val_x = torch.stack(val_x).transpose(0, 1)
    val_y = all_data_val[0][-1]

    index = [_ for _ in range(len(val_x))]
    random.shuffle(index)
    train_index = index[:int(len(index) * 0.8)]
    val_index = index[int(len(index) * 0.8):]

    train_x, train_y = val_x[train_index].to(device).to(torch.float), val_y[train_index].to(device)
    val_x, val_y = val_x[val_index].to(device).to(torch.float), val_y[val_index].to(device)

    test_x = torch.stack(test_x).transpose(0, 1).to(device).to(torch.float)
    label = all_data[0][-1].to(device)
    model = MyModel(val_x.shape[-1], task).to(device)
    train(model, train_x, train_y, val_x, val_y, task)
    out, _ = model(test_x, label)
    if task == 'TASK1':
        preds = out.argmax(-1).to('cpu').numpy()
    else:
        preds = (out > 0).to(torch.long).to('cpu').numpy()
    return preds

