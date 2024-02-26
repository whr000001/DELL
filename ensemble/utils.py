import os
from openai import OpenAI
from tenacity import retry, wait_fixed, stop_after_attempt
import tiktoken

os.environ['http_proxy'] = 'http://127.0.0.1:15236'
os.environ['https_proxy'] = 'http://127.0.0.1:15236'

client = OpenAI(
        api_key='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'
    )
encoder = tiktoken.encoding_for_model("gpt-3.5-turbo-instruct")


@retry(wait=wait_fixed(1), stop=stop_after_attempt(3))
def get_reply(prompt, return_logprobs=False):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=100,
        logprobs=5
    )
    logprobs = response.choices[0].logprobs
    out = {
        'text_offset': logprobs.text_offset,
        'token_logprobs': logprobs.token_logprobs,
        'tokens': logprobs.tokens,
        'top_logprobs': logprobs.top_logprobs,
    }
    if return_logprobs:
        return response.choices[0].text, out
    return response.choices[0].text


def construct_length(text, length=3840):
    codes = encoder.encode(text)
    codes = codes[:length]
    return encoder.decode(codes)
