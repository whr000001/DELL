import json
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class MySampler(Sampler):
    def __init__(self, indices, shuffle):
        super().__init__(None)
        self.indices = indices
        if not torch.is_tensor(self.indices):
            self.indices = torch.tensor(self.indices, dtype=torch.long)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = self.indices[torch.randperm(self.indices.shape[0])]
        else:
            indices = self.indices
        for item in indices:
            yield item

    def __len__(self):
        return len(self.indices)


TASK2_label = {
    'MFC': ['Morality', 'Policy Prescription and Evaluation', 'Crime and Punishment', 'Health and Safety',
            'Security and Defense', 'Economic', 'Political', 'Public Sentiment', 'Cultural Identity',
            'Capacity and Resources primany', 'Quality of life', 'Legality, Constitutionality, Jurisdiction',
            'Policy Presecription and Evaluation', 'Capacity and Resources',
            'Fairness and Equality', 'External regulation and reputation', 'Quality of Life',
            'External Regulation and Reputation'],
    'SemEval-23F': ['Security_and_defense', 'Fairness_and_equality', 'Political', 'Capacity_and_resources',
                    'Economic', 'Morality', 'Policy_prescription_and_evaluation',
                    'Legality_Constitutionality_and_jurisprudence', 'External_regulation_and_reputation',
                    'Quality_of_life', 'Health_and_safety', 'Cultural_identity', 'Crime_and_punishment',
                    'Public_opinion']
}


TASK3_label = {
    'Generated': ['flag waving', 'straw man', 'false dilemma', 'name calling', 'guilt by association',
                  'whataboutism', 'repetition', 'obfuscation', 'slogans', 'conversation killer', 'appeal to hypocrisy',
                  'doubt', 'loaded language', 'exaggeration', 'causal oversimplification', 'appeal to fear',
                  'appeal to popularity', 'red herring', 'appeal to authority'],
    'SemEval-20': ['Appeal_to_fear-prejudice', 'Black-and-White_Fallacy', 'Name_Calling,Labeling', 'Slogans',
                   'Whataboutism,Straw_Men,Red_Herring', 'Exaggeration,Minimisation', 'Loaded_Language',
                   'Repetition', 'Causal_Oversimplification', 'Bandwagon,Reductio_ad_hitlerum', 'Flag-Waving',
                   'Thought-terminating_Cliches', 'Appeal_to_Authority', 'Doubt'],
    'SemEval-23P': ['Conversation_Killer', 'False_Dilemma-No_Choice', 'Appeal_to_Popularity', 'Doubt', 'Flag_Waving',
                    'Slogans', 'Whataboutism', 'Straw_Man', 'Loaded_Language', 'Name_Calling-Labeling',
                    'Obfuscation-Vagueness-Confusion', 'Appeal_to_Fear-Prejudice', 'Causal_Oversimplification',
                    'Red_Herring', 'Repetition', 'Exaggeration-Minimisation', 'Appeal_to_Authority',
                    'Guilt_by_Association', 'Appeal_to_Hypocrisy']
}


class NewsDataset(Dataset):
    def __init__(self, task, dataset_name, root='../data'):
        data = json.load(open(f'{root}/datasets/{task}/{dataset_name}.json'))

        # preprocess the label of each instance
        labels = []
        if task == 'TASK1':
            for item in data:
                labels.append(item[1])
            self.num_class = 2
        else:
            if task == 'TASK2':
                label_list = TASK2_label[dataset_name]
            else:
                label_list = TASK3_label[dataset_name]
            label_dict = {item: index for index, item in enumerate(label_list)}
            for item in data:
                this_label = item[1]
                if not isinstance(this_label, list):
                    this_label = [this_label]
                out_label = [0 for _ in range(len(label_list))]
                for _ in this_label:
                    out_label[label_dict[_]] = 1
                labels.append(out_label)
            self.num_class = len(labels[0])

        #  load the output of each proxy task
        #  content-based proxy task
        sentiment = json.load(open(f'{root}/proxy/sentiment/{task}_{dataset_name}.json'))
        framing = json.load(open(f'{root}/proxy/framing/{task}_{dataset_name}.json'))
        propaganda = json.load(open(f'{root}/proxy/propaganda/{task}_{dataset_name}.json'))
        retrieval = json.load(open(f'{root}/proxy/retrieval/{task}_{dataset_name}.json'))

        #  comment/graph-based proxy task
        graph_info = []
        for index in range(len(data)):
            comment = json.load(open(f'{root}/networks/{task}_{dataset_name}/{index}.json'))
            stance = json.load(open(f'{root}/proxy/stance/{task}_{dataset_name}/{index}.json'))
            response = json.load(open(f'{root}/proxy/response/{task}_{dataset_name}/{index}.json'))
            #  load the graph structure
            row, col = [], []
            for tgt, src in enumerate(comment['father']):
                if src is None:
                    continue
                row.append(src)
                col.append(tgt)
            edge_index = [row, col]
            comment = comment['res'][1:]
            response = response[1:]
            stance = stance[1:]
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            graph_info.append({
                'comment': comment,
                'stance': stance,
                'relation': response,
                'edge_index': edge_index
            })

        self.data = []
        for index in range(len(data)):
            self.data.append({
                'text': data[index][0],
                'sentiment': sentiment[index],
                'framing': framing[index],
                'propaganda': propaganda[index],
                'retrieval': retrieval[index],
                'graph_info': graph_info[index],
                'label': labels[index],
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def my_collate_fn(batch):
    text = []
    sentiment = []
    framing = []
    propaganda = []
    retrieval = []
    label = []
    graph_info = []
    for item in batch:
        text.append(item['text'])
        sentiment.append(item['sentiment'])
        framing.append(item['framing'])
        propaganda.append(item['propaganda'])
        retrieval.append(item['retrieval'])
        label.append(item['label'])
        graph_info.append(item['graph_info'])
    label = torch.tensor(label, dtype=torch.long)
    return {
        'text': text,
        'emotion': sentiment,
        'framing': framing,
        'propaganda': propaganda,
        'retrieval': retrieval,
        'label': label,
        'graph_info': graph_info,
        'batch_size': len(batch),
    }


def main():
    dataset = NewsDataset('TASK1', 'LLM-mis')
    loader = DataLoader(dataset, batch_size=4, collate_fn=my_collate_fn)
    for batch in loader:
        print(batch.keys())
        print(batch['sentiment'][0])
        print(batch['label'].shape)
        input()


if __name__ == '__main__':
    main()
