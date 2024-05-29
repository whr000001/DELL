import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import nltk


def get_reply(content):
    return 'this is an example output'


# #  load LLMs, we employ mistral-7b as LLM, from https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1",
#                                              # local_files_only=True,
#                                              device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", local_files_only=True)
#
#
# @torch.no_grad()
# def get_reply(content):
#     sentences = [content]
#     inputs = tokenizer(sentences, return_tensors="pt")
#     inputs = inputs.to(model.device)
#     input_length = len(inputs['input_ids'][0])
#     generated_ids = model.generate(**inputs,
#                                    max_new_tokens=1000,
#                                    temperature=0,
#                                    do_sample=True,
#                                    pad_token_id=tokenizer.eos_token_id)
#     decoded = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)
#     return decoded


def construct_length(text, length=10000):
    if len(text) < length:
        return text
    sents = nltk.sent_tokenize(text)
    out = ''
    for sent in sents:
        if len(out) + len(sent) + 1 <= length:
            out = out + ' ' + sent
        else:
            break
    return out


def sentiment_detection(text):
    prompt = f'News:\n{text}\nQuestion:\n'
    prompt += 'Which emotions does the news contain? ' \
              'Please choose the three most likely ones: anger, disgust, fear, happiness, sadness, and surprise. '
    prompt += 'Please provide your reasoning.'
    prompt += '\nAnswer:'
    res = get_reply(prompt)
    return res


def framing_detection(text):
    prompt = f'News:\n{text}\nTask:\n'
    prompt += 'Framing is a strategic device and a central concept in political communication ' \
              'for representing different salient aspects and perspectives to convey the latent meaning of an issue. '
    prompt += 'Which framings does the news contain? Please choose the five most likely ones: '
    prompt += 'Economic; Capacity and resources; Morality; Fairness and equality; ' \
              'Legality, constitutionality and jurisprudence; Policy prescription and evaluation; ' \
              'Crime and punishment; Security and defense; Health and safety; Quality of life; Cultural identity;' \
              ' Among public opinion; Political; External regulation and reputation. '
    prompt += 'Please provide your reasoning.'
    prompt += '\nAnswer:'
    res = get_reply(prompt)
    return res


def propaganda_detection(text):
    prompt = f'News:\n{text}\nTask:\n'
    prompt += 'Propaganda tactics are methods used in propaganda to convince an audience ' \
              'to believe what the propagandist wants them to believe. '
    prompt += 'Which propaganda techniques does the news contain? Please choose the five most likely ones: '
    prompt += 'Conversation Killer; Whataboutism; Doubt; Straw Man; Red Herring; Loaded Language; ' \
              'Appeal to Fear-Prejudice; Guilt by Association; Flag Waving; False Dilemma-No Choice; ' \
              'Repetition; Appeal to Popularity; Appeal to Authority; Name Calling-Labeling; Slogans; ' \
              'Appeal to Hypocrisy; Exaggeration-Minimisation; Obfuscation-Vagueness-Confusion; ' \
              'Causal Oversimplification. '
    prompt += 'Please provide your reasoning.'
    prompt += '\nAnswer:'
    res = get_reply(prompt)
    return res


def find_entity(text):
    prompt = f'News:\n{text}\n'
    prompt += 'TASK:\n'
    prompt += 'Identify five named entities within the news above that necessitate elucidation ' \
              'for the populace to understand the news comprehensively. '
    prompt += 'Ensure a diverse selection of the entities. The answer should in the form of python list.'
    prompt += '\nAnswer:'
    res = get_reply(prompt)
    return res


def get_stance(textA, textB):
    textA = construct_length(textA, length=5000)
    textB = construct_length(textB, length=5000)
    prompt = 'TASK:\nDetermine the stance of sentence 2 on sentence 1. ' \
             'Is it supportive, neutral or opposed? Provide your reasoning.\n'
    prompt += f'Sentence 1: {textA}\n'
    prompt += f'Sentence 2: {textB}\n'
    prompt += 'Answer:'
    res = get_reply(prompt)
    return res


def get_response(textA, textB):
    textA = construct_length(textA, length=5000)
    textB = construct_length(textB, length=5000)
    prompt = f'Sentence 1:\n{textA}\n'
    prompt += f'Sentence 2:\n{textB}\n'
    prompt += 'TASK:\n'
    prompt += 'Sentence 1 and Sentence 2 are two posts on social networks. '
    prompt += 'Please judge whether the sentence 2 replies to the sentence 1. '
    prompt += 'Answer yes or no and provide the reasoning.\n'
    prompt += 'Answer:'
    res = get_reply(prompt)
    return res
