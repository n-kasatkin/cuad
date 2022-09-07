#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=3,
export PYTHONPATH=.:${PYTHONPATH}


taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 python3 train.py \
        --output_dir ./output/roberta-base \
        --cache_dir ./output/cache \
        --model_type roberta \
        --model_name_or_path roberta-base \
        --train_file ./data/train_separate_questions.json \
        --predict_file ./data/test.json \
        --do_train \
        --do_eval \
        --version_2_with_negative \
        --learning_rate 1e-4 \
        --num_train_epochs 4 \
        --per_gpu_eval_batch_size=12  \
        --per_gpu_train_batch_size=12 \
        --max_seq_length 512 \
        --max_answer_length 512 \
        --doc_stride 256 \
        --save_steps 1000 \
        --n_best_size 20 \
        --overwrite_output_dir \
        --num_workers 4 \
        --threads 8 \
        --logging_steps=1 \
