#/bin/bash
TASK=./data_path/
DICT="./pt_model/gpt2bpe/dict.txt"
cd ./fairseq
for LANG in src tgt
do
for SPLIT in train dev
do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json ./pt_model/gpt2bpe/encoder.json \
    --vocab-bpe ./pt_model/gpt2bpe/vocab.bpe \
    --inputs "$TASK/$SPLIT.$LANG" \
    --outputs "$TASK/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty
done
done

fairseq-preprocess \
  --source-lang src \
  --target-lang tgt \
  --trainpref ${TASK}/train.bpe \
  --validpref ${TASK}/dev.bpe \
  --destdir ${TASK}/bin \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --tgtdict ${DICT} \
  --workers 70