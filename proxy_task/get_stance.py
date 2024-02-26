import os
import json
from tqdm import tqdm
from utils import generate_res


def get_one_stance(data):
    prompt = 'TASK:\nDetermine the stance of sentence 2 on sentence 1. Is it supportive, neutral or opposed? Provide your reasoning.\n'
    prompt += 'Sentence 1: {}\n'
    prompt += 'Sentence 2: {}\n'
    prompt += 'Answer:\n'
    # data = json.load(open('comment_tree/TASK1_F3/0.json'))
    father = data['father']
    res = data['res']
    stances = []
    for index, item in enumerate(res):
        if father[index] is None:
            stances.append(None)
        else:
            source = res[father[index]]
            target = res[index]
            source = source[:2000]
            target = target[:2000]
            question = prompt.format(source, target)
            out = generate_res(question)
            stances.append(out)
    assert len(stances) == len(res)
    return stances


def main():
    datasets = os.listdir('comment_tree')
    for dataset in datasets:
        path = 'comment_tree/{}'.format(dataset)
        if not os.path.exists('stance/{}'.format(dataset)):
            os.mkdir('stance/{}'.format(dataset))
        files = os.listdir(path)
        for file in tqdm(files):
            save_path = 'stance/{}/{}'.format(dataset, file)
            if os.path.exists(save_path):
                continue
            data = json.load(open('{}/{}'.format(path, file)))
            data_res = get_one_stance(data)
            json.dump(data_res, open(save_path, 'w'))


if __name__ == '__main__':
    main()
