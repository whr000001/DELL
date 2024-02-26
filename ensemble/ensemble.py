import json
import os
import torch
# from utils import get_reply, construct_length
from parameter import dataset_names
from tqdm import tqdm
from sklearn.metrics import f1_score, jaccard_score
import re
from pred_utils import likelihood_weight
import numpy as np
# from mistral_utils import get_reply, construct_length


def get_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro') * 100, \
        f1_score(y_true, y_pred, average='micro') * 100, \
        jaccard_score(y_true, y_pred, average='macro') * 100


def run(task, dataset, run_type='confidence'):
    save_dir = 'mistral_out/{}_{}_{}.json'.format(task, dataset, run_type)
    data = json.load(open('datasets/{}/{}.json'.format(task, dataset)))
    test_indices = json.load(open('split/{}_{}/test.json'.format(task, dataset)))

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
        expert = torch.load('results/{}_{}_{}_{}.pt'.format(task, dataset, text, graph))
        all_experts.append(expert)
    if os.path.exists(save_dir):
        out = json.load(open(save_dir))
    else:
        out = []
    for _, index in enumerate(tqdm(test_indices[len(out):])):
        news = construct_length(data[index][0].strip())
        prompt = 'News:\n{}\n\n'.format(news)
        # prompt += description.data[task][dataset]
        if run_type == 'confidence':
            prompt += 'Some experts give predictions and confidence scores about the news. '
            prompt += 'The higher the score, the more confidence the result is.\n'
        else:
            prompt += 'Some experts give predictions about the news.\n'
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

            expert_info += 'The expert predicts the label of this news is {}. '.format(expert_ans)
            if run_type == 'confidence':
                if task == 'TASK1':
                    expert_info += 'The confidence scores are {:.2f}.\n'.format(expert_likelihood)
                else:
                    expert_info += 'The confidence scores are {}.\n'.format(expert_likelihood)
            else:
                expert_info += '\n'
            prompt += expert_info
        prompt += '\n'
        prompt += 'Question:\nBased on the analysis of experts, please judge the final label of this news. '
        prompt += 'Give the label in the form of \"[your answer]\", do not give any explanation.\n'
        prompt += 'Label:\n'
        res = get_reply(prompt)
        out.append(res)
        # res, confidence = get_reply(prompt)
        # out.append([res, confidence])
        json.dump(out, open(save_dir, 'w'))
        # input()


def find_bracketed_substrings(input_string):
    # 使用正则表达式查找被方括号包裹的部分
    pattern = r'\[(.*?)\]'
    res = re.findall(pattern, input_string)
    return res[0]


def evaluate(task, dataset, run_type):
    data = torch.load('results/{}_{}_None_None.pt'.format(task, dataset))
    label = data[-1].numpy().tolist()
    out = json.load(open('mistral_out/{}_{}_{}.json'.format(task, dataset, run_type)))
    default_preds = likelihood_weight(task, dataset).tolist()
    preds = []
    for index, item in enumerate(out):
        if isinstance(item, list):
            item = item[0]
        if task != 'TASK1':
            try:
                pred = find_bracketed_substrings(item)
                pred = [int(_) for _ in pred.split()]
                pred = [1 if _ != 0 else 0 for _ in pred]
                assert len(pred) == len(label[0])
            except Exception as e:
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
    for task in ['TASK1', 'TASK2', 'TASK3']:
        for dataset in dataset_names[task]:
            # run(task, dataset, run_type='confidence')
            # run(task, dataset, run_type='x')
            evaluate(task, dataset, run_type='x')
            evaluate(task, dataset, run_type='confidence')


if __name__ == '__main__':
    main()
