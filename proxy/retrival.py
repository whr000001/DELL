import json
import nltk


dataset_names = {
    'TASK1': ['LLM-mis', 'Pheme'],
    'TASK2': ['MFC', 'SemEval-23F'],
    'TASK3': ['Generated', 'SemEval-20', 'SemEval-23P']
}


def main():
    for task in dataset_names:
        for dataset in dataset_names[task]:
            data = json.load(open(f'../data/datasets/{task}/{dataset}.json'))
            wiki = json.load(open(f'../data/proxy/entity_from_wiki/{task}_{dataset}.json'))
            save_dir = f'../data/proxy/retrieval/{task}_{dataset}.json'
            out = []
            for item, each_wiki in zip(data, wiki):
                out_text = item[0]
                for entity, exp in each_wiki:
                    if exp is None:
                        continue
                    exp = ' '.join(nltk.sent_tokenize(exp)[:3])
                    entity_index = out_text.lower().find(entity.lower())
                    if entity_index != -1:
                        entity_index = entity_index + len(entity)
                        out_text = out_text[:entity_index] + ' (' + exp + ')' + out_text[entity_index:]
                out.append(out_text)
            json.dump(out, open(save_dir, 'w'))


if __name__ == '__main__':
    main()
