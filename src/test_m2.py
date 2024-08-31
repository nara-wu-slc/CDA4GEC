import sys
sys.path.append('./')
from m2scorer_python3 import m2scorer
from m2scorer_python3.scripts.Tokenizer import PTBTokenizer
import re

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
    max_unchanged_words=2
    beta = 0.5
    ignore_whitespace_casing= False
    verbose = True
    very_verbose = False
    
    system_file_in = './system.in'
    system_file_out = './system.out'
    gold_file = './data/conll14/conll14.m2'

    # load source sentences and gold edits
    source_sentences, gold_edits = m2scorer.load_annotation(gold_file)

    # load system hypotheses
    fin = m2scorer.smart_open(system_file_in, 'r')
    fout = m2scorer.smart_open(system_file_out, 'r')

    system_sentences = [clean(line_out.strip()) for line_out in fout.readlines()][:1312]
        
    fin.close()
    fout.close()
    
    p, r, f1 = m2scorer.levenshtein.batch_multi_pre_rec_f1(system_sentences, source_sentences, gold_edits, max_unchanged_words, beta, ignore_whitespace_casing, verbose, very_verbose)

    print(("Precision   : %.4f" % p))
    print(("Recall      : %.4f" % r))
    print(("F_%.1f       : %.4f" % (beta, f1)))

