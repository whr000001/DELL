import json
import os
import random
from tqdm import tqdm
from utils import generate_res


def get_relation(a, b):
    prompt = 'Sentence 1:\n{}\n'.format(a)
    prompt += 'Sentence 2:\n{}\n'.format(b)
    prompt += 'TASK:\n'
    prompt += 'Sentence 1 and Sentence 2 are two posts on social networks. '
    prompt += 'Please judge whether the sentence 2 replies to the sentence 1. '
    prompt += 'Answer yes or no and provide the reasoning.\n'
    prompt += 'Answer:\n'
    res = generate_res(prompt)
    return res


def main():
    datasets = os.listdir('comment_tree')
    datasets = sorted(datasets)
    for dataset in datasets:
        files = os.listdir('comment_tree/{}'.format(dataset))
        pbar = tqdm(files, leave=False, desc=dataset)
        save_dir = 'relation/{}'.format(dataset)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for file in pbar:
            save_path = '{}/{}'.format(save_dir, file)
            data = json.load(open('comment_tree/{}/{}'.format(dataset, file), encoding='utf-8'))
            father = data['father']
            res = data['res']
            edges = []
            for tgt, src in enumerate(father):
                if src is None:
                    continue
                edges.append([src, tgt])
            positive = edges
            negative = []
            for item in positive:
                src = item[0]
                tgt = random.sample([_ for _ in range(len(father)) if [src, _] not in positive and _ != src], k=1)[0]
                negative.append([src, tgt])
            out = {'positive': positive, 'negative': negative}
            if os.path.exists(save_path):
                continue
            ans = []
            for item in positive + negative:
                tmp = get_relation(res[item[0]], res[item[1]])
                ans.append(tmp)
            out['ans'] = ans
            json.dump(out, open(save_path, 'w', encoding='utf-8'))


if __name__ == '__main__':
    main()
