#!/bin/bash

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

python3 run_qa_no_trainer.py \
  --model_name_or_path bert-base-chinese \
  --train_file ./train.json \
  --validation_file ./valid.json \
  --output_dir output/run_qa_no_trainer/ \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --max_seq_length 512 \
  --num_train_epochs 1 \
  --learning_rate 3e-5