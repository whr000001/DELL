import os
import json
from tqdm import tqdm
from utils import get_stance


dataset_names = {
    'TASK1': ['LLM-mis', 'Pheme'],
    'TASK2': ['MFC', 'SemEval-23F'],
    'TASK3': ['Generated', 'SemEval-20', 'SemEval-23P']
}


def get_one_stance(data):
    father = data['father']
    res = data['res']
    stances = []
    for index, item in enumerate(res):
        if father[index] is None:
            stances.append(None)
        else:
            source = res[father[index]]
            target = res[index]
            out = get_stance(source, target)
            stances.append(out)
    assert len(stances) == len(res)
    return stances


def main():
    for task in dataset_names:
        for dataset in dataset_names[task]:
            network_dir = f'../data/networks/{task}_{dataset}'
            save_dir = f'../data/proxy/stance/{task}_{dataset}'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            files = os.listdir(network_dir)
            for file in tqdm(files):
                save_path = f'{save_dir}/{file}'
                if os.path.exists(save_path):
                    continue
                data = json.load(open(f'{network_dir}/{file}'))
                res = get_one_stance(data)
                json.dump(res, open(save_path, 'w'))


if __name__ == '__main__':
    main()
