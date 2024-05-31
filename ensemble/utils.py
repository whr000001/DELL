import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


#  load LLMs, we employ mistral-7b as LLM, from https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
#  you can change it to other LLMs such as ChatGPT or Llama
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1",
                                             # local_files_only=True,
                                             device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", local_files_only=True)


@torch.no_grad()
def get_reply(content):
    sentences = [content]
    inputs = tokenizer(sentences, return_tensors="pt")
    inputs = inputs.to(model.device)
    input_length = len(inputs['input_ids'][0])
    generated_ids = model.generate(**inputs,
                                   max_new_tokens=1000,
                                   temperature=0.2,
                                   do_sample=True,
                                   pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)
    return decoded

