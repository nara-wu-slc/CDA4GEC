import errant
import spacy
import json
from tqdm import tqdm
import sys
sys.path.append('./')
from m2scorer_python3.scripts.Tokenizer import PTBTokenizer
from torch.utils.data import DataLoader
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str)
parser.add_argument("--output_path", type=str)
args= parser.parse_args()

def get_edits(wrong, right):
    orig = annotator.parse(wrong)
    cor = annotator.parse(right)
    edits = annotator.annotate(orig, cor)
    return [(e.o_start, e.o_end, e.o_str, e.c_start, e.c_end, e.c_str, e.type) for e in edits]


def clean(text):
    tokens = m2_tokenizer.tokenize(text)
    out = ' '.join(tokens)
    for rule in retokenization_rules:
        out = re.sub(rule[0], rule[1], out)
    return out

if __name__ == '__main__':
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
    
    nlp = spacy.load('en_core_web_sm')
    annotator = errant.load('en', nlp)

    text = json.load(open(args.file_path, 'r'))

    for item in tqdm(text):
        item['input'] = item['input'].strip()
        item['output'] = item['output'].strip()
        if item['input'] == item['output']:
            item['edits'] = []
        else:
            item['edits'] = get_edits(item['input'], item['output'])
    
    json.dump(text, open(args.output_path, 'w'), indent=2, ensure_ascii=False)    