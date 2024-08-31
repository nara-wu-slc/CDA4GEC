from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel
)
import torch
import datasets
import json
from tqdm import tqdm
import sys
sys.path.append('./')
from m2scorer_python3.scripts.Tokenizer import PTBTokenizer
from torch.utils.data import DataLoader
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpt_model_path", type=str)
parser.add_argument("--template_path", type=str)
parser.add_argument("--output_path", type=str)
args= parser.parse_args()

if __name__ == '__main__':
    model_path = args.gpt_model_path
    data_path = args.template_path
    model = GPT2LMHeadModel.from_pretrained(model_path)
    t = GPT2Tokenizer.from_pretrained(model_path)
    ds = datasets.load_dataset('json', data_files=data_path)['train']
    dl = DataLoader(ds, shuffle=False, batch_size=64)
    res_list = []
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
    model.cuda()
    with torch.no_grad():
        for text in tqdm(dl):
            input_batch = t([x + '<sep>' for x in text['input']], return_tensors='pt', padding=True, max_length=256)
            res = model.generate(input_ids=input_batch.input_ids.cuda(),
                                 attention_mask=input_batch.attention_mask.cuda(),
                                max_length=128,
                                # num_beams=5,
                                do_sample=True,
                                top_k=50,
                                top_p=0.95,
                                temperature=0.7,
                                repetition_penalty=1.2
                                )
            res = t.batch_decode(res)
            for idx, n in enumerate(res):
                n = n.split('<sep>')[-1].strip()
                n = n.replace('<|endoftext|>', '')
                def clean(text):
                    tokens = m2_tokenizer.tokenize(text)
                    out = ' '.join(tokens)
                    for rule in retokenization_rules:
                        out = re.sub(rule[0], rule[1], out)
                    return out
                
                wrong_sent = ' '.join(m2_tokenizer.tokenize(n.strip()))
                for rule in retokenization_rules:
                    wrong_sent = re.sub(rule[0], rule[1], wrong_sent)
                correct_sent = wrong_sent
                
                right_list = re.sub(r'<mask>', '\n', text['input'][idx]).strip().split('\n')
                wrong_list = re.sub(r'<mask>', '\n', text['wrong'][idx]).strip().split('\n')
                total = 0
                skip = 0
                for i in range(len(right_list)):
                    total += 1
                    if clean(right_list[i]).strip() in wrong_sent:
                        wrong_sent = wrong_sent.replace(clean(right_list[i]).strip(), clean(wrong_list[i]).strip(), 1)
                    else:
                        skip += 1
                    print(skip/total)
                
                res_list.append({
                        'input': wrong_sent,
                        'output': correct_sent,
                    })

    with open(f'{args.output_path}/train.src', 'w') as fsrc:
        with open(f'{args.output_path}/train.tgt', 'w') as ftgt:
            for item in res_list:
                fsrc.write(item['input']+'\n')
                ftgt.write(item['output']+'\n')