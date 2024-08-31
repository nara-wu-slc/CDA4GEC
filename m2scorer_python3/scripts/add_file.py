from optparse import OptionParser

from m2scorer_python3.scripts.util import uniq
import re
import sys
from copy import deepcopy
from m2scorer_python3.scripts.Tokenizer import PTBTokenizer
from m2scorer_python3.scripts.levenshtein import(
    levenshtein_matrix, 
    edit_graph, 
    merge_graph,
    transitive_arcs,
    best_edit_seq_bf,
    set_weights,
    shrinkEdit,
)
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

def pre_treatment(sent):
    tokens = m2_tokenizer.tokenize(sent.strip())
    out = ' '.join(tokens)
    for rule in retokenization_rules:
        out = re.sub(rule[0], rule[1], out)
    return out


def get_m2_list(source_sent, target_sent, gold_sent, max_unchanged_words=2, beta=0.5, ignore_whitespace_casing= False, verbose=False, very_verbose=False):
    source_sent = pre_treatment(source_sent).split()
    target_sent = pre_treatment(target_sent).split()
    gold_sent = pre_treatment(gold_sent).split()

    lmatrix1, backpointers1 = levenshtein_matrix(source_sent, target_sent, 1, 1, 1)
    lmatrix2, backpointers2 = levenshtein_matrix(source_sent, target_sent, 1, 1, 2)
    lmatrix3, backpointers3 = levenshtein_matrix(source_sent, gold_sent, 1, 1, 1)
    lmatrix4, backpointers4 = levenshtein_matrix(source_sent, gold_sent, 1, 1, 2)

    V1, E1, dist1, edits1 = edit_graph(lmatrix1, backpointers1)
    V3, E3, dist3, edits3 = edit_graph(lmatrix3, backpointers3)
    V2, E2, dist2, edits2 = edit_graph(lmatrix2, backpointers2)
    V4, E4, dist4, edits4 = edit_graph(lmatrix4, backpointers4)

    V, E, dist, edits = merge_graph(V1, V2, E1, E2, dist1, dist2, edits1, edits2)
    Vg, Eg, distg, editsg = merge_graph(V3, V4, E3, E4, dist3, dist4, edits3, edits4)

    V, E, dist, edits = transitive_arcs(V, E, dist, edits, max_unchanged_words, very_verbose)
    Vg, Eg, distg, editsg = transitive_arcs(Vg, Eg, distg, editsg, max_unchanged_words, very_verbose)

    localdistg = set_weights(Eg, distg, editsg, edits, verbose, very_verbose)
    editSeqg = best_edit_seq_bf(Vg, Eg, localdistg, editsg, very_verbose)

    localdist = set_weights(E, dist, edits, editsg, verbose, very_verbose)
    editSeq = best_edit_seq_bf(V, E, localdist, edits, very_verbose)

    editSeq = [shrinkEdit(ed) for ed in list(reversed(editSeq))]
    editSeqG = [shrinkEdit(ed) for ed in list(reversed(editSeqg))]
    
    return editSeq, editSeqG


if __name__ == '__main__':
    res = get_m2_list("As the development of the technology , social media becomes more and more significant role in the whole world .",
                       "As a result of the development of technology , social media plays a more and more significant role in the whole world .",
                       "As a result of the development of technology , social media plays a more and more significant role in the whole world .")
    print(res)