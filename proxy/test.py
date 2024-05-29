import json
import nltk


def main():
    dataset_names = {
        'TASK1': ['LLM-mis', 'Pheme'],
        'TASK2': ['MFC', 'SemEval-23F'],
        'TASK3': ['Generated', 'SemEval-20', 'SemEval-23P']
    }
    transfer_dict = {
        ('TASK1', 'LLM-mis'): 'TASK1_LLM-misinformation',
        ('TASK2', 'MFC'): 'TASK2_MFC',
        ('TASK2', 'SemEval-23F'): 'TASK2_semeval-2023',
        ('TASK3', 'Generated'): 'TASK3_generated',
        ('TASK3', 'SemEval-20'): 'TASK3_semeval-2020',
        ('TASK3', 'SemEval-23P'): 'TASK3_semeval-2023',
    }
    # for task in dataset_names:
    #     for dataset in dataset_names[task]:
    #         save_dir = f'../data/proxy/entity_from_wiki/{task}_{dataset}.json'
    #         if (task, dataset) not in transfer_dict:
    #             continue
    #         transfer_name = transfer_dict[(task, dataset)]
    #         data_dir = f'../../LLMs&misinformation/LLMs/DATA/{transfer_name}_12345.json'
    #         data = json.load(open(data_dir))
    #         out = []
    #         for item in data:
    #             out.append(item)
    #         json.dump(out, open(save_dir, 'w'))

    for task in dataset_names:
        for dataset in dataset_names[task]:
            save_dir = f'../data/proxy/response/{task}_{dataset}'
            if (task, dataset) not in transfer_dict:
                continue
            transfer_name = transfer_dict[(task, dataset)]
            data_dir = f'../../LLMs&misinformation/LLMs/DATA/'

    # entity_wiki = json.load(open('../../LLMs&misinformation/LLMs/generate_prompt/entity_wiki.json'))
    # res_entity = json.load(open('../../LLMs&misinformation/LLMs/generate_prompt/res/find_entity.json', encoding='utf-8'))
    #
    # dataset_names = {
    #     'TASK1': ['LLM-misinformation', 'LLM-misinfo-QA', 'F3'],
    #     'TASK2': ['MFC', 'semeval-2023'],
    #     'TASK3': ['generated', 'semeval-2020', 'semeval-2023']
    # }
    # cnt = 0
    # for TASK in dataset_names:
    #     for dataset_name in dataset_names[TASK]:
    #         out = []
    #         data = json.load(open('../../LLMs&misinformation/LLMs/datasets/{}/{}.json'.format(TASK, dataset_name), encoding='utf-8'))
    #         for index in range(len(data)):
    #             res = res_entity[cnt]
    #             wiki = entity_wiki[cnt]
    #             text = data[index][0]
    #
    #             entities = res.split(']')[0]
    #             entities = entities[1:]
    #             entities = entities.split(',')
    #             entities = [_.strip()[1:-1] for _ in entities]
    #             xxx = []
    #             for entity, exp in zip(entities, wiki):
    #                 xxx.append([entity, exp])
    #             out.append(xxx)
    #             cnt += 1
    #         json.dump(out, open(f'../../LLMs&misinformation/LLMs/DATA/{TASK}_{dataset_name}_12345.json', 'w'))


    # entity_wiki = json.load(open('../../LLMs&misinformation/LLMs/generate_prompt/pheme_entity_wiki.json'))
    # res_entity = json.load(open('../../LLMs&misinformation/LLMs/generate_prompt/Pheme_wiki.json', encoding='utf-8'))
    #
    # out = []
    # data = json.load(
    #     open('../../LLMs&misinformation/LLMs/datasets/TASK1/Pheme.json', encoding='utf-8'))
    # cnt = 0
    # for index in range(len(data)):
    #     res = res_entity[cnt]
    #     wiki = entity_wiki[cnt]
    #     text = data[index][0]
    #
    #     entities = res.split(']')[0]
    #     entities = entities[1:]
    #     entities = entities.split(',')
    #     entities = [_.strip()[1:-1] for _ in entities]
    #     xxx = []
    #     for entity, exp in zip(entities, wiki):
    #         xxx.append([entity, exp])
    #     out.append(xxx)
    #     cnt += 1
    # json.dump(out, open('../data/proxy/entity_from_wiki/Pheme.json', 'w'))


if __name__ == '__main__':
    main()
