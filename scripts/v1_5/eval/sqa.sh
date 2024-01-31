#!/bin/bash

#CKPT="7b-vision_chat-ft-1M-v2-1-imageft-1"
CKPT="7b-laion-2B-en-imageft-1"
python -m llava.eval.model_vqa_science \
    --model-path /mnt/disks/disk-1/checkpoints/converted/$CKPT \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode lvm

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT}_result.json
