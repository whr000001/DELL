import os
import torch
import json
from dataset import NewsDataset, my_collate_fn, MySampler
from model import MyModel
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro') * 100


@torch.no_grad()
def validation(model, loader, epoch, task):
    model.eval()
    all_truth = []
    all_preds = []
    pbar = tqdm(loader, desc='valuate {} epoch'.format(epoch), leave=False)
    save_out = []
    save_pred = []
    save_label = []
    for batch in pbar:
        out, _ = model(batch)
        if task == 'TASK1':
            preds = out.argmax(-1).to('cpu')
        else:
            preds = (out > 0).to(torch.long).to('cpu')
        save_out.append(out.to('cpu'))
        save_pred.append(preds.to('cpu'))
        save_label.append(batch['label'].to('cpu'))
        # print(out, preds)
        # input()
        truth = batch['label'].to('cpu')
        all_truth.append(truth)
        all_preds.append(preds)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    save_out = torch.cat(save_out, dim=0)
    save_pred = torch.cat(save_pred, dim=0)
    save_label = torch.cat(save_label, dim=0)
    return get_metric(all_truth, all_preds), save_out, save_pred, save_label


def main():
    # modify task and dataset_name
    task = 'TASK3'
    dataset_name = 'SemEval-23P'
    # modify the expert
    text_augmentation = None
    graph_augmentation = 'relation'

    # load the checkpoints that achieves the best validation performance
    checkpoints_dir = 'checkpoints/{}_{}_{}_{}'.format(task, dataset_name, text_augmentation, graph_augmentation)
    checkpoints = os.listdir(checkpoints_dir)
    checkpoint_path = sorted(checkpoints, reverse=True)[0]
    print('{}_{}_{}_{}: '.format(task, dataset_name, text_augmentation, graph_augmentation), checkpoint_path)
    checkpoint = torch.load('{}/{}'.format(checkpoints_dir, checkpoint_path))

    dataset = NewsDataset(task, dataset_name)
    test_indices = json.load(open(f'../data/split/{task}_{dataset_name}/test.json'))
    val_indices = json.load(open(f'../data/split/{task}_{dataset_name}/val.json'))
    # It is recommended to set it the same as during training
    batch_size = 1

    test_sampler = MySampler(test_indices, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate_fn, sampler=test_sampler)
    val_sampler = MySampler(val_indices, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate_fn, sampler=val_sampler)
    model = MyModel(num_class=test_loader.dataset.num_class, task=task,
                    device=device, lm_path='microsoft/deberta-v3-large',
                    text_augmentation=text_augmentation,
                    graph_augmentation=graph_augmentation).to(device)

    if not os.path.exists('results'):
        os.mkdir('results')
    model.load_state_dict(checkpoint)
    metric, out, pred, label = validation(model, test_loader, 0, task)
    print('test: ', metric)
    torch.save([out, pred, label], f'results/{task}_{dataset_name}_{text_augmentation}_{graph_augmentation}_test.pt')

    metric, out, pred, label = validation(model, val_loader, 0, task)
    print('val: ', metric)
    torch.save([out, pred, label], f'results/{task}_{dataset_name}_{text_augmentation}_{graph_augmentation}_val.pt')


if __name__ == '__main__':
    main()
