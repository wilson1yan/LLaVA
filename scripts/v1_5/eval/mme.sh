#!/bin/bash

CKPT="7b-laion-2B-en-imageft-1"
#python -m llava.eval.model_vqa_loader \
#    --model-path /mnt/disks/disk-1/checkpoints/converted/$CKPT \
#    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
#    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
#    --answers-file ./playground/data/eval/MME/answers/$CKPT.jsonl \
#    --temperature 0 \
#    --conv-mode lvm

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool

python calculation.py --results_dir answers/$CKPT
