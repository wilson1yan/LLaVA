#!/bin/bash

NAME=1vm-1epoch

#python -m llava.eval.model_vqa \
#    --model-path /home/wilsonyan/checkpoints/converted/7b-vision-laion-coco-webvid-mmc4-2-shuffle-imageft-1-1epoch \
#    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
#    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
#    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$NAME.jsonl \
#    --temperature 0 \
#    --conv-mode lvm

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/$NAME.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/$NAME.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/$NAME.jsonl
