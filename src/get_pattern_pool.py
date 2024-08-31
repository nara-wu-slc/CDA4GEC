import json
from tqdm import tqdm
from collections import Counter, defaultdict
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file_with_error", type=str)
parser.add_argument("--sample_num", type=int, default=10000)
parser.add_argument("--min_count", type=int, default=2)
parser.add_argument("--gram", type=int, default=1)
parser.add_argument("--output_path", type=str)
args= parser.parse_args()

if __name__ == '__main__':
    train_file = json.load(open(args.file_with_error, 'r'))

    pool = []
    pool_1gram = []
    pool_1gram_map = defaultdict(list)
    count = args.sample_num
    min_count = args.min_count
    gram = args.gram
    for item in tqdm(train_file):
        for case in item['edits']:
            ei_back   = (case[2], case[5])
            sent = item['output'].split(' ')
            if case[2] == '' or case[5] == '':
                start = max(0, case[3]-gram)
                end = min(len(sent), case[4]+gram)
                for idx in range(case[3]-1, start-1, -1):
                    case[2] = (sent[idx] + ' ' + case[2]).strip()
                    case[5] = (sent[idx] + ' ' + case[5]).strip()
                for idx in range(case[4], end):
                    case[2] = (case[2] + ' ' + sent[idx]).strip()
                    case[5] = (case[5] + ' ' + sent[idx]).strip()
            ei = (case[2], case[5]) + ei_back
            pool.append(ei)
            pool_1gram.append(ei_back)
    
    pool_1gram_count = dict(Counter(pool_1gram))
    error_item = [x for x in set(pool) if pool_1gram_count[x[2:]]>=min_count]
    # print(len(error_item))

    for ei in error_item:
        pool_1gram_map[ei[2:]].append(ei[:2])
    
    # select = []
    select = defaultdict(int)
    res_list = []
    for idx in tqdm(range(count)):
        sample_num = random.choices([1, 2, 3], weights=[1, 2, 1], k=1)[0]
        instruct = []
        wrong_word = []
        pattern = []
        for i in range(sample_num):
            main_class = random.choice(list(pool_1gram_map.keys()))
            sub_class = random.choice(pool_1gram_map[main_class])
            pattern.append(sub_class)
            select[str(main_class)] += 1
        
        flag = False
        for num in range(sample_num):
            instruct += ["<mask>"]
            wrong_word += ["<mask>"]

            if pattern[num][0] != '':
                wrong_word += [pattern[num][0]]
            else:
                continue

            if pattern[num][1] != '':
                instruct += [pattern[num][1]]
            else:
                flag = True
        
        if not flag:
            res_list.append({'input':' '.join(instruct[1:]), 'wrong': ' '.join(wrong_word[1:])})
    
    json.dump(select, open(f'{args.output_path}/select_pool.json', 'w'), indent=2)
    
    json.dump(res_list, open(f'{args.output_path}/template.json', 'w'), indent=2, ensure_ascii=False)