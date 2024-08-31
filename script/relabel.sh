#!/bin/bash

model_dir=./output/your_model_path

fairseq-interactive ./data/conll14/bin \
  --path $model_dir/checkpoint_best.pt \
  --task translation_for_gec \
  --gen-subset valid \
  --bpe 'gpt2' \
  --gpt2-encoder-json ./pt_model/gpt2bpe/encoder.json \
  --gpt2-vocab-bpe ./pt_model/gpt2bpe/vocab.bpe \
  --nbest 1 \
  --beam 12 \
  --sacrebleu \
  --max-len-b	256\
  --max-tokens 4096 \
  --buffer-size 10000 \
  --input ./data/pseudo.src > pseudo.log

cat pseudo.log | grep "^D-"  | python -c "import sys; x = sys.stdin.readlines(); x = [ x[i] for i in range(len(x)) ]; x = ''.join(x) ; print(x)" | cut -f 3 > pseudo.out