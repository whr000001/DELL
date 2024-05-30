import json
import nltk
import torch


def main():
    dataset_names = {
        'TASK1': ['LLM-mis', 'Pheme'],
        'TASK2': ['MFC', 'SemEval-23F'],
        'TASK3': ['Generated', 'SemEval-20', 'SemEval-23P']
    }
    transfer_dict = {
        ('TASK1', 'Pheme'): 'TASK1_Pheme',
        ('TASK1', 'LLM-mis'): 'TASK1_LLM-misinformation',
        ('TASK2', 'MFC'): 'TASK2_MFC',
        ('TASK2', 'SemEval-23F'): 'TASK2_semeval-2023',
        ('TASK3', 'Generated'): 'TASK3_generated',
        ('TASK3', 'SemEval-20'): 'TASK3_semeval-2020',
        ('TASK3', 'SemEval-23P'): 'TASK3_semeval-2023',
    }
    for task in dataset_names:
        for dataset in dataset_names[task]:
            if (task, dataset) not in transfer_dict:
                continue
            transfer_name = transfer_dict[(task, dataset)]
            text_augmentation = ['None', 'emotion', 'framing', 'propaganda', 'retrieval', 'None', 'None']
            graph_augmentation = ['None', 'None', 'None', 'None', 'None', 'stance', 'relation']
            for text, graph in zip(text_augmentation, graph_augmentation):
                data = torch.load(f'../../LLMs&misinformation/ensemble_final/results/{transfer_name}_{text}_{graph}.pt')
                save_dir = f'results/{task}_{dataset}_{text}_{graph}_test.pt'
                torch.save(data, save_dir)

                data = torch.load(f'../../LLMs&misinformation/ensemble_final/results/{transfer_name}_{text}_{graph}_val.pt')
                save_dir = f'results/{task}_{dataset}_{text}_{graph}_val.pt'
                torch.save(data, save_dir)
            # data_dir = f'../../LLMs&misinformation/LLMs/DATA/{transfer_name}_12345.json'
            # data = json.load(open(data_dir))
            # out = []
            # for item in data:
            #     out.append(item)
            # json.dump(out, open(save_dir, 'w'))



if __name__ == '__main__':
    main()
