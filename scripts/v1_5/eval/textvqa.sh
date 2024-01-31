#!/bin/bash

#CKPT="7b-vision_chat-ft-1M-v2-1-imageft-1"
CKPT="7b-laion-2B-en-imageft-1"
python -m llava.eval.model_vqa_loader \
    --model-path /mnt/disks/disk-1/checkpoints/converted/$CKPT \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode lvm

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl
