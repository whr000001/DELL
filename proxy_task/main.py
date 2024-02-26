import json
import os.path
from utils import generate_res
from tqdm import tqdm


dataset_names = {
    'TASK1': ['LLM-misinformation', 'Pheme'],
    'TASK2': ['MFC', 'semeval-2023'],
    'TASK3': ['generated', 'semeval-2020', 'semeval-2023']
}


def run_emotion_detection():
    prompt = 'News:\n{}\nQuestion:\n'
    prompt += 'Which emotions does the news contain? Please choose the three most likely ones: anger, disgust, fear, happiness, sadness, and surprise. '
    prompt += 'Please provide your reasoning.'
    prompt += '\nAnswer:\n'
    for task in dataset_names:
        for dataset in dataset_names[task]:
            data = json.load(open('datasets/{}/{}.json'.format(task, dataset)))
            save_dir = 'DATA/{}_{}_emotion_detection.json'.format(task, dataset)
            if os.path.exists(save_dir):
                out = json.load(open(save_dir))
            else:
                out = []
            for item in tqdm(data[len(out):]):
                input_text = item[0][:2000]
                question = prompt.format(input_text)
                res = generate_res(question)
                out.append(res)
                json.dump(out, open(save_dir, 'w'))


def run_framing_detection():
    prompt = 'News:\n{}\nTask:\n'
    prompt += 'Framing is a strategic device and a central concept in political communication for representing different salient aspects and perspectives to convey the latent meaning of an issue. '
    prompt += 'Which framings does the news contain? Please choose the five most likely ones: '
    prompt += 'Economic; Capacity and resources; Morality; Fairness and equality; Legality, constitutionality and jurisprudence; Policy prescription and evaluation; Crime and punishment; Security and defense; Health and safety; Quality of life; Cultural identity; Among public opinion; Political; External regulation and reputation. '
    prompt += 'Please provide your reasoning.'
    prompt += '\nAnswer:\n'
    for task in dataset_names:
        for dataset in dataset_names[task]:
            data = json.load(open('datasets/{}/{}.json'.format(task, dataset)))
            save_dir = 'DATA/{}_{}_framing_detection.json'.format(task, dataset)
            if os.path.exists(save_dir):
                out = json.load(open(save_dir))
            else:
                out = []
            for item in tqdm(data[len(out):]):
                input_text = item[0][:2000]
                question = prompt.format(input_text)
                res = generate_res(question)
                out.append(res)
                json.dump(out, open(save_dir, 'w'))


def run_propaganda_detection():
    prompt = 'News:\n{}\nTask:\n'
    prompt += 'Propaganda techniques are methods used in propaganda to convince an audience to believe what the propagandist wants them to believe. '
    prompt += 'Which propaganda techniques does the news contain? Please choose the five most likely ones: '
    prompt += 'Conversation Killer; Whataboutism; Doubt; Straw Man; Red Herring; Loaded Language; Appeal to Fear-Prejudice; '
    prompt += 'Guilt by Association; Flag Waving; False Dilemma-No Choice; Repetition; Appeal to Popularity; Appeal to Authority; '
    prompt += 'Name Calling-Labeling; Slogans; Appeal to Hypocrisy; Exaggeration-Minimisation; Obfuscation-Vagueness-Confusion; Causal Oversimplification. '
    prompt += 'Please provide your reasoning.'
    prompt += '\nAnswer:\n'
    for task in dataset_names:
        for dataset in dataset_names[task]:
            data = json.load(open('datasets/{}/{}.json'.format(task, dataset)))
            save_dir = 'DATA/{}_{}_propaganda_detection.json'.format(task, dataset)
            if os.path.exists(save_dir):
                out = json.load(open(save_dir))
            else:
                out = []
            for item in tqdm(data[len(out):]):
                input_text = item[0][:2000]
                question = prompt.format(input_text)
                res = generate_res(question)
                out.append(res)
                json.dump(out, open(save_dir, 'w'))


def run_find_entity():
    prompt = 'News:\n{}\n'
    prompt += 'TASK:\n'
    prompt += 'Identify five named entities within the news above that necessitate elucidation for the populace to understand the news comprehensively. '
    prompt += 'Ensure a diverse selection of the entities. The answer should in the form of python list.'
    prompt += 'Answer:\n'

    for task in dataset_names:
        for dataset in dataset_names[task]:
            data = json.load(open('datasets/{}/{}.json'.format(task, dataset)))
            save_dir = 'entity/{}_{}.json'.format(task, dataset)
            if os.path.exists(save_dir):
                out = json.load(open(save_dir))
            else:
                out = []
            # print(len(data))
            for item in tqdm(data[len(out):]):
                input_text = item[0][:2000]
                question = prompt.format(input_text)
                res = generate_res(question)
                out.append(res)
                json.dump(out, open(save_dir, 'w'))


def main():
    run_emotion_detection()
    run_framing_detection()
    run_propaganda_detection()
    run_find_entity()


if __name__ == '__main__':
    main()
