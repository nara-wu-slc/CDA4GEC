#!/bin/bash

model_dir=./output/your_model_path

fairseq-generate ./data/conll14/bin \
  --path $model_dir/checkpoint_best.pt \
  --task translation_for_gec \
  --gen-subset valid \
  --bpe 'gpt2' \
  --gpt2-encoder-json ./pt_model/gpt2bpe/encoder.json \
  --gpt2-vocab-bpe ./pt_model/gpt2bpe/vocab.bpe \
  --nbest 1 \
  --beam 12 \
  --sacrebleu \
  --batch-size 32 > generator.log

cat generator.log | grep "^D-"  | python -c "import sys; x = sys.stdin.readlines(); x = [ x[i] for i in range(len(x)) if (i % 1 == 0) ]; x = sorted(x, key=lambda x:int(x.split('\t')[0][2:])) ; x = ''.join(x) ; print(x)" | cut -f 3 > system.out
cat generator.log | grep "^S-"  | python -c "import sys; x = sys.stdin.readlines(); x = [ x[i] for i in range(len(x)) if (i % 1 == 0) ]; x = sorted(x, key=lambda x:int(x.split('\t')[0][2:])) ; x = ''.join(x) ; print(x)" | cut -f 2 > system.in

python src/test_m2.py