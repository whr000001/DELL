import json
import os
import torch
from utils import get_reply
from tqdm import tqdm
from sklearn.metrics import f1_score
import re
from ensemble_utils import likelihood_weight
import numpy as np


dataset_names = {
    'TASK1': ['LLM-mis', 'Pheme'],
    'TASK2': ['MFC', 'SemEval-23F'],
    'TASK3': ['Generated', 'SemEval-20', 'SemEval-23P']
}


def get_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro') * 100, \
        f1_score(y_true, y_pred, average='micro') * 100


def run(task, dataset):
    if not os.path.exists('results'):
        os.mkdir('results')
    save_dir = f'results/confidence_{task}_{dataset}.json'
    data = json.load(open(f'../data/datasets/{task}/{dataset}.json'))
    test_indices = json.load(open(f'../data/split/{task}_{dataset}/test.json'))

    # experts and description of each expert
    text_augmentation = ['None', 'emotion', 'framing', 'propaganda', 'retrieval', 'None', 'None']
    graph_augmentation = ['None', 'None', 'None', 'None', 'None', 'stance', 'relation']

    expert_descriptions = [
        'This expert is comprehensive. ',
        'This expert focuses on the emotion of this news. ',
        'This expert focuses on the framing of this news. ',
        'This expert focuses on the propaganda technology of this news. ',
        'This expert focuses on the external knowledge of this news. ',
        'This expert focuses on the stance of related comments on this news. ',
        'This expert focuses on the relation of related comments on this news. '
    ]

    all_experts = []
    for text, graph in zip(text_augmentation, graph_augmentation):
        # load prediction of each expert
        expert = torch.load(f'../expert/results/{task}_{dataset}_{text}_{graph}_test.pt')
        all_experts.append(expert)
    if os.path.exists(save_dir):
        out = json.load(open(save_dir))
    else:
        out = []
    for _, index in enumerate(tqdm(test_indices[len(out):])):
        news = data[index][0].strip()
        # truncate news article if needed
        prompt = 'News:\n{}\n\n'.format(news)
        prompt += 'Some experts give predictions and confidence scores about the news. '
        prompt += 'The higher the score, the more confidence the result is.\n'
        for _index in range(len(all_experts)):
            expert = all_experts[_index]
            expert_ans = expert[1][_]
            expert_ans = expert_ans.data.numpy()

            if task != 'TASK1':
                expert_likelihood = expert[0][_]
                expert_likelihood = torch.abs(expert_likelihood).data.numpy()
            else:
                expert_likelihood = expert[0][_]
                expert_likelihood = torch.softmax(expert_likelihood, dim=-1)
                expert_likelihood = torch.max(expert_likelihood, dim=-1)[0]
                expert_likelihood = torch.abs(expert_likelihood).data.numpy()
            expert_likelihood = np.round(expert_likelihood, 2)

            expert_info = 'Expert {}:\n'.format(_index + 1)
            expert_info += expert_descriptions[_index]

            expert_info += f'The expert predicts the label of this news is {expert_ans}. '
            if task == 'TASK1':
                expert_info += 'The confidence scores are {:.2f}.\n'.format(expert_likelihood)
            else:
                expert_info += 'The confidence scores are {}.\n'.format(expert_likelihood)
            prompt += expert_info
        prompt += '\n'
        prompt += 'Question:\nBased on the analysis of experts, please judge the final label of this news. '
        prompt += 'Give the label in the form of \"[your answer]\", do not give any explanation.\n'
        prompt += 'Label:'
        res = get_reply(prompt)
        out.append(res)
        json.dump(out, open(save_dir, 'w'))


def find_bracketed_substrings(input_string):
    pattern = r'\[(.*?)\]'
    res = re.findall(pattern, input_string)
    return res[0]


def evaluate(task, dataset):
    data = torch.load('../expert/results/{}_{}_None_None_test.pt'.format(task, dataset))
    label = data[-1].numpy().tolist()
    results = json.load(open(f'results/confidence_{task}_{dataset}.json'))
    default_preds = likelihood_weight(task, dataset).tolist()
    preds = []
    for index, item in enumerate(results):
        if isinstance(item, list):
            item = item[0]
        if task != 'TASK1':
            try:
                pred = find_bracketed_substrings(item)
                pred = [int(_) for _ in pred.split()]
                pred = [1 if _ != 0 else 0 for _ in pred]
                assert len(pred) == len(label[0])
            except Exception:
                pred = default_preds[index]
            preds.append(pred)
        else:
            if '1' in item:
                preds.append(1)
            elif '0' in item:
                preds.append(0)
            else:
                preds.append(default_preds[index])
    print(get_metric(label, preds))


def main():
    for task in dataset_names:
        for dataset in dataset_names[task]:
            run(task, dataset)
            evaluate(task, dataset)


if __name__ == '__main__':
    main()
