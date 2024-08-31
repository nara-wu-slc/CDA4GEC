import random
import sys
sys.path.append('./')
from m2scorer_python3.scripts.Tokenizer import PTBTokenizer
from vllm import LLM, SamplingParams
import re, json
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--llama_model_path", type=str)
parser.add_argument("--template_path", type=str)
parser.add_argument("--output_path", type=str)
args= parser.parse_args()

def get_prompt(text):
    prompts =[
    "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
    "Use phrases from #input to make sentences."
    "You should fill in <mask> to make #input sentence more complete."
    "You can't change any form or order of the words in #input."
    "Make sure you fully use the phrases in #input."
    "[/INST]"
    "#input:\n <mask> sized city with eighty thousand <mask>\n"
    "#output:\n My town is a medium - sized city with eighty thousand inhabitants .\n\n"
    "#input:\n <mask> my own plan too , <mask> to be the same as them . <mask>\n"
    "#output:\n I have my own plan too , but I do n't want to be the same as them . I want to become a journalist .\n\n"
    "#input:\n Nowadays , each family has more than 1 <mask> one of several reasons why <mask>\n"
    "#output:\n Nowadays , each family has more than 1 car for each person , this is only one of several reasons why people use less public transport .\n\n</s>"
    "#input:\n <mask> they might want to safeguard <mask>\n"
    "#output:\n On the other hand , they might want to safeguard the national image .\n\n"
    "#input:\n Lucy , Molly , and <mask> a cowboy , and a <mask>\n"
    "#output:\n Lucy , Molly , and their parents , a cowboy , and a teacher .\n\n"
    f"#input:\n{text}\n"
    ]
    return prompts[0]

sampling_params = SamplingParams(n=5, temperature=0.9, top_p=0.95, max_tokens=512, stop='\n\n')


llm = LLM(model=args.llama_model_path, seed=417)

generate_text = []
with open(args.template_path) as f:
    for item in f:
        generate_text.append(json.loads(item))
pre_input = [get_prompt(x['input']) for x in generate_text]

outputs = llm.generate(pre_input, sampling_params)

m2_tokenizer = PTBTokenizer()
retokenization_rules = [
    # Remove extra space around single quotes, hyphens, and slashes.
    (" ' (.*?) ' ", " '\\1' "),
    (" - ", "-"),
    (" / ", "/"),
    # Ensure there are spaces around parentheses and brackets.
    (r"([\]\[\(\){}<>])", " \\1 "),
    (r"\s+", " "),
]
def clean(text):
    tokens = m2_tokenizer.tokenize(text)
    out = ' '.join(tokens)
    for rule in retokenization_rules:
        out = re.sub(rule[0], rule[1], out)
    return out

def check_count(token, sentence):
    a = 0
    t = 0
    for item in token:
        if clean(item.strip()) in sentence:
            a += 1
        t += 1
    return a, t

res_list = []

acc = 0
total = 0
for input_text, output in zip(generate_text, outputs):
    prompt = output.prompt
    generated_text = output.outputs
    input_token = re.sub('<mask>', '\n', input_text['input']).strip().split('\n')
    wrong_token = re.sub('<mask>', '\n', input_text['wrong']).strip().split('\n')
    temp_acc = -1
    temp_tt = -1
    temp_sent = ''
    for res_text in generated_text:
        if "#output:\n" not in res_text.text:
            continue
        res_sentence = clean(res_text.text.split('\n')[-1].strip())
        aa, tt = check_count(input_token, res_sentence)
        if aa > temp_acc:
            temp_acc = aa
            temp_tt = tt
            temp_sent = res_sentence
    
    acc += temp_acc
    total += temp_tt
    
    wrong_sent = temp_sent
    correct_sent = temp_sent
    
    for c, w in zip(input_token, wrong_token):
        wrong_sent = wrong_sent.replace(clean(c).strip(), clean(w).strip(), 1)
    
    if random.random() > 0.5:
        res_list.append({
            'input': wrong_sent,
            'output': correct_sent
        })
    else:
        res_list.append({
            'input': correct_sent,
            'output': correct_sent
        })

print(acc)
print(total)

with open(f'{args.output_path}/train.src', 'w') as fs:
    with open(f'{args.output_path}/train.tgt', 'w') as ft:
        for item in res_list:
            fs.write(item['input']+'\n')
            ft.write(item['output']+'\n')
    