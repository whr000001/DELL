import os
import json
from tqdm import tqdm
from utils import sentiment_detection, framing_detection, propaganda_detection, find_entity


dataset_names = {
    'TASK1': ['LLM-mis', 'Pheme'],
    'TASK2': ['MFC', 'SemEval-23F'],
    'TASK3': ['Generated', 'SemEval-20', 'SemEval-23P']
}


def main():
    for task in dataset_names:
        for dataset in dataset_names[task]:
            data = json.load(open(f'../data/datasets/{task}/{dataset}.json'))
            #  run the sentiment analysis proxy task
            if not os.path.exists('../data'):
                os.mkdir('../data')
            if not os.path.exists('../data/proxy'):
                os.mkdir('../data/proxy')
            if not os.path.exists('../data/proxy/sentiment'):
                os.mkdir('../data/proxy/sentiment')
            save_dir = f'../data/proxy/sentiment/{task}_{dataset}.json'
            if os.path.exists(save_dir):
                out = json.load(open(save_dir))
            else:
                out = []
            for item in tqdm(data[len(out):], leave=False, desc=f'{task}_{dataset}_sentiment'):
                news_article = item[0]
                #  truncate news article if needed
                sentiment = sentiment_detection(news_article)
                out.append(sentiment)
                json.dump(out, open(save_dir, 'w'))

            #  run the framing detection proxy task
            if not os.path.exists('../data/proxy/framing'):
                os.mkdir('../data/proxy/framing')
            save_dir = f'../data/proxy/framing/{task}_{dataset}.json'
            if os.path.exists(save_dir):
                out = json.load(open(save_dir))
            else:
                out = []
            for item in tqdm(data[len(out):], leave=False, desc=f'{task}_{dataset}_framing'):
                news_article = item[0]
                #  truncate news article if needed
                framing = framing_detection(news_article)
                out.append(framing)
                json.dump(out, open(save_dir, 'w'))

            #  run the propaganda tactics detection proxy task
            if not os.path.exists('../data/proxy/propaganda'):
                os.mkdir('../data/proxy/propaganda')
            save_dir = f'../data/proxy/propaganda/{task}_{dataset}.json'
            if os.path.exists(save_dir):
                out = json.load(open(save_dir))
            else:
                out = []
            for item in tqdm(data[len(out):], leave=False, desc=f'{task}_{dataset}_propaganda'):
                news_article = item[0]
                #  truncate news article if needed
                propaganda = propaganda_detection(news_article)
                out.append(propaganda)
                json.dump(out, open(save_dir, 'w'))

            #  find the helpful entities for understanding the news
            if not os.path.exists('../data/proxy/entity'):
                os.mkdir('../data/proxy/entity')
            save_dir = f'../data/proxy/entity/{task}_{dataset}.json'
            if os.path.exists(save_dir):
                out = json.load(open(save_dir))
            else:
                out = []
            for item in data[len(out):]:
                news_article = item[0]
                #  truncate news article if needed
                entities = find_entity(news_article)
                out.append(entities)
                json.dump(out, open(save_dir, 'w'))


if __name__ == '__main__':
    main()
