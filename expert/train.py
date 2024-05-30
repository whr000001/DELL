import json
from dataset import NewsDataset, my_collate_fn, MySampler
from model import MyModel
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
from sklearn.metrics import f1_score
from argparse import ArgumentParser


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--batch_size', type=int, default='4')
parser.add_argument('--text_augmentation', type=str)
parser.add_argument('--graph_augmentation', type=str)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()

task = args.task
dataset_name = args.dataset_name
batch_size = args.batch_size
text_augmentation = args.text_augmentation
graph_augmentation = args.graph_augmentation
lr = args.lr


def get_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro') * 100


def train_one_epoch(model, optimizer, loader, epoch):
    model.train()
    ave_loss = 0
    cnt = 0
    all_truth = []
    all_preds = []
    pbar = tqdm(loader, desc='train {} epoch'.format(epoch), leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        out, loss = model(batch)
        if task == 'TASK1':
            preds = out.argmax(-1).to('cpu')
        else:
            preds = (out > 0).to(torch.long).to('cpu')
        truth = batch['label'].to('cpu')
        loss.backward()
        optimizer.step()
        ave_loss += loss.item() * len(batch)
        cnt += len(batch)
        all_truth.append(truth)
        all_preds.append(preds)
    ave_loss /= cnt
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    return ave_loss, get_metric(all_truth, all_preds)


@torch.no_grad()
def validation(model, loader, epoch):
    model.eval()
    all_truth = []
    all_preds = []
    pbar = tqdm(loader, desc='valuate {} epoch'.format(epoch), leave=False)
    for batch in pbar:
        out, _ = model(batch)
        if task == 'TASK1':
            preds = out.argmax(-1).to('cpu')
        else:
            preds = (out > 0).to(torch.long).to('cpu')
        truth = batch['label'].to('cpu')
        all_truth.append(truth)
        all_preds.append(preds)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    return get_metric(all_truth, all_preds)


def train(train_loader, val_loader, test_loader, name):
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    save_path = 'checkpoints/{}'.format(name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model = MyModel(num_class=train_loader.dataset.num_class, task=task,
                    device=device, lm_path='microsoft/deberta-v3-large',
                    text_augmentation=text_augmentation,
                    graph_augmentation=graph_augmentation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_acc = 0
    best_state = model.state_dict()
    for key, value in best_state.items():
        best_state[key] = value.clone()
    no_up = 0
    for _ in range(100):
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, _)
        acc = validation(model, val_loader, _)
        if acc > best_acc:
            best_acc = acc
            no_up = 0
            for key, value in model.state_dict().items():
                best_state[key] = value.clone()
        else:
            if _ >= 8:
                no_up += 1
        if no_up >= 16:
            break
        print('train loss: {:.2f}, train metric: {:.2f}, now best val metric: {:.2f}'.
              format(train_loss, train_acc, best_acc))
    model.load_state_dict(best_state)
    acc = validation(model, test_loader, 0)
    print('train {} done. val metric: {:.2f}, test metric: {:.2f}'.format(name, best_acc, acc))
    cnt = 0
    while True:
        checkpoint_path = '{}/{:.2f}_{}.pt'.format(save_path, acc, cnt)
        if not os.path.exists(checkpoint_path):
            break
        cnt += 1
    torch.save(best_state, checkpoint_path)


def main():
    dataset = NewsDataset(task, dataset_name)
    train_indices = json.load(open(f'../data/split/{task}_{dataset_name}/train.json'))
    val_indices = json.load(open(f'../data/split/{task}_{dataset_name}/val.json'))
    test_indices = json.load(open(f'../data/split/{task}_{dataset_name}/test.json'))

    train_sampler = MySampler(train_indices, shuffle=True)
    val_sampler = MySampler(val_indices, shuffle=False)
    test_sampler = MySampler(test_indices, shuffle=False)
    train_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate_fn, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate_fn, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate_fn, sampler=test_sampler)
    train_name = '{}_{}_{}_{}'.format(task, dataset_name, text_augmentation, graph_augmentation)
    for _ in range(5):
        train(train_loader, val_loader, test_loader, train_name)


if __name__ == '__main__':
    main()
