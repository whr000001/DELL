import os
import random
import json
import torch
from tqdm import tqdm
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import nltk


#  load LLMs, we employ mistral-7b as LLM, from https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 5, 7'
model = AutoModelForCausalLM.from_pretrained("/data01/whr/resources/Mistral-7B-Instruct-v0.3",
                                             device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("/data01/whr/resources/Mistral-7B-Instruct-v0.3")


def construct_length(text):
    sents = nltk.sent_tokenize(text)
    out = ''
    for sent in sents:
        if len(out) + len(sent) + 1 <= 640:
            out = out + ' ' + sent
        else:
            break
    return out


@torch.no_grad()
def get_reply(content, temperature=0.3):
    sentences = [content]
    inputs = tokenizer(sentences, return_tensors="pt")
    inputs = inputs.to(model.device)
    input_length = len(inputs['input_ids'][0])
    if temperature == 0:
        generated_ids = model.generate(**inputs,
                                       max_new_tokens=200,
                                       do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id)
    else:
        generated_ids = model.generate(**inputs,
                                       max_new_tokens=200,
                                       temperature=temperature,
                                       do_sample=True,
                                       pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)
    return decoded


characters = {
    'gender': ['male', 'female'],
    'age': ['0-17', '18-29', '30-49', '50-64', '65+'],
    'ethnicity': ['White', 'Black', 'Hispanic'],
    'education': ['college grad', 'some college', 'HS or less'],
    'income': ['75000 or more', '30000-74999', 'less than 30000'],
    'party': ['Republican', 'Democrat'],
    'vote': ['registered to vote', 'probably to vote', 'not registered']
}


profile_text = {
    'gender': ['You are male.', 'You are female.'],
    'age': ['You are under 17 years old.', 'You are 18 to 29 years old.', 'You are 30 to 49 years old.',
            'You are 50 to 64 years old.', 'You are over 65 years old.'],
    'ethnicity': ['Racially, you are {}.'.format(_) for _ in ['White', 'Black', 'Hispanic']],
    'income': ['Financially, your annual family income is {}.'.format(_)
               for _ in ['more than 75,000', '30,000 to 74,999', 'less than 30,000']],
    'education': ['Educationally, you {}.'.format(_)
                  for _ in ['are a college grad', 'haven\'t graduated from college',
                            'have a high school diploma or less']],
    'party': ['Politically, you are a {}.'.format(_)
              for _ in ['Democrat', 'Republican']],
    'vote': ['Meanwhile, you are {}.'.format(_)
             for _ in ['registered to vote', 'probably registered to vote', 'not registered to vote']]
}


def generate_a_character():
    profile = 'You are a social media user. '
    for item in profile_text:
        profile += random.choice(profile_text[item]) + ' '
    return profile


def get_news_comment(user, news):
    prompt = '{}\nYou view a piece of news with the following content.\n'.format(user)
    prompt += 'News:\n{}\n'.format(news)
    prompt += 'Task:\nPlease comment on this news on social media. Your comment is limited to 40 words. '
    prompt += '\nYour comment:'
    res = get_reply(prompt)
    return res


def get_comment_comment(user, news, comments):
    prompt = '{}\nYou view a piece of news and a related comment chain on social media, ' \
             'and their contents are as follows.\n'.format(user)
    prompt += 'News:\n{}\n'.format(news)
    for index, comment in enumerate(comments):
        prompt += 'Comment {}: {}\n'.format(index+1, comment)
    prompt += 'Task:\nPlease reply to the last comment(comment {}) on social media. '.format(len(comments))
    prompt += 'Your reply is limited to 40 words. '
    prompt += '\nYour reply:'
    res = get_reply(prompt)
    return res


def get_comment_comment_chain(user, news, comment_chains):
    prompt = '{}\nYou view a piece of news and related comment chains on social media, ' \
             'and their contents are as follows.\n'.format(user)
    prompt += 'News:\n{}\n'.format(news)
    for chain_index, comments in enumerate(comment_chains):
        prompt += 'Comment Chain {}:\n'.format(chain_index+1)
        for index, comment in enumerate(comments):
            prompt += 'Comment {}: {}\n'.format(index+1, comment)
    prompt += 'Task:\nPlease select a comment chain that you would most like to comment. ' \
              'Answer selected number and explain the reason.\n'
    prompt += 'Answer:'
    res = get_reply(prompt)
    return res


def generate(news):
    def get_chain(node_id):
        chain_id = [node_id]
        while True:
            node_id = father[node_id]
            if node_id == 0:
                break
            chain_id.append(node_id)
        chain_id = list(reversed(chain_id))
        return [res[_] for _ in chain_id]

    def comment_comment(func_candidate):
        if len(func_candidate) == 1:
            comment = get_chain(func_candidate[0])
            res.append(get_comment_comment(user, res[0], comment))
            return func_candidate[0], None
        else:
            chain = [get_chain(_) for _ in func_candidate]
            func_reason = get_comment_comment_chain(user, res[0], chain)
            try:
                match = re.search(r'\d+', func_reason)
                choice_id = int(match.group(0)) - 1
                choice = chain[choice_id]
            except Exception as e:
                choice_id = random.choice(range(len(chain)))
                func_reason = str(e)
                choice = chain[choice_id]
            res.append(get_comment_comment(user, res[0], choice))
            return func_candidate[choice_id], func_reason

    torch.cuda.empty_cache()

    res = [news]
    users = []
    reasons = []

    size_limit = 11  # parameter to control the network size
    rt_ratio = 0.2  # parameter to control the probability of commenting on the news article, namely alpha in our paper
    height_ratio = 0.6  # parameter to balance the tree height and width, namely bata in our paper
    eps = 1e-5
    k = 3  # parameter to control candidate set size

    father = [None]
    height = [0]
    width = [0]
    graph_size = 1
    rt = 0

    while graph_size < size_limit:
        user = generate_a_character()
        users.append(user)
        if graph_size == 1:
            graph_size += 1
            father.append(rt)
            height.append(height[rt] + 1)
            width.append(0)
            res.append(get_news_comment(user, res[rt]))
            width[rt] += 1
            reasons.append(None)
        else:
            identify = random.uniform(0, 1)
            if identify <= rt_ratio:
                graph_size += 1
                father.append(rt)
                height.append(height[rt] + 1)
                width.append(0)
                res.append(get_news_comment(user, res[rt]))
                width[rt] += 1
                reasons.append(None)
            else:
                candidate = [_ for _ in range(1, graph_size)]
                probabilities = [height[_] * height_ratio + width[_] * (1 - height_ratio) + eps
                                 for _ in candidate]
                probabilities = np.array(probabilities)
                probabilities /= np.sum(probabilities)
                candidate_comment = np.random.choice(candidate, p=probabilities,
                                                     size=min(k, len(candidate)), replace=False)
                fa, reason = comment_comment(candidate_comment)
                reasons.append(reason)

                graph_size += 1
                father.append(fa)
                height.append(height[fa] + 1)
                width.append(0)
                width[fa] += 1
    father = [int(_) if _ is not None else None for _ in father]
    return {
        'father': father,  # the father node id of each node
        'users': users,  # the user profile of each node
        'res': res,  # the comment content of each node
        'reasons': reasons  # the LLMs' generated reason
    }


def main():
    random.seed(20250101)
    dataset_names = {
        'TASK1': ['LLM-mis', 'Pheme'],
        'TASK2': ['MFC', 'SemEval-23F'],
        'TASK3': ['Generated', 'SemEval-20', 'SemEval-23P']
    }
    for task in dataset_names:
        for dataset in dataset_names[task]:
            dataset_name = dataset.replace('.json', '')
            if not os.path.exists('../data/networks'):
                os.mkdir('../data/networks')
            save_dir = f'../data/networks/{task}_{dataset}'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            data = json.load(open(f'../data/datasets/{task}/{dataset}.json', encoding='utf-8'))
            #  load datasets
            for index, item in enumerate(tqdm(data, desc='{}, {}'.format(task, dataset_name), leave=False)):
                save_path = f'{save_dir}/{index}.json'
                if os.path.exists(save_path):
                    continue
                #  a way to limit the length of news articles, optiona
                in_text = construct_length(item[0])
                #  obtain the user-news network. If you want to generate user-news networks on other datasets,
                #  you can use the generate function.
                out = generate(in_text)  # the input is news article
                json.dump(out, open(save_path, 'w', encoding='utf-8'))  # save the network


if __name__ == '__main__':
    main()
