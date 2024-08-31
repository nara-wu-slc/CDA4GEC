#!/bin/bash
PRETRAIN=./pt_model/bart_large/model.pt

fairseq-train ./data/xxxx/bin \
  --arch bart_large \
  --task translation_for_gec \
  --save-dir ./output/c4200m_pt_new\
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr 3e-05 --max-epoch 10 \
  --max-tokens 8192 --update-freq 8 \
  --keep-last-epochs 10 \
  --seed 47 --log-format tqdm --log-interval 2 \
  --restore-file	$PRETRAIN \
  --fp16 \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \