#!/bin/bash

SPLIT="mmbench_dev_20230712"
CKPT="7b-vision_chat-ft-1M-v2-1-imageft-1"

python -m llava.eval.model_vqa_mmbench \
    --model-path /mnt/disks/disk-1/checkpoints/converted/$CKPT \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode lvm

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $CKPT
