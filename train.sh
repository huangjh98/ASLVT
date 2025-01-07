#!/bin/bash
DATA_PATH="./dataset/USRD"
python -m torch.distributed.launch --nproc_per_node=1 \
--use_env main_task_retrieval.py --do_test --num_thread_reader=4 \
--epochs=5 --batch_size=64 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/Videos \
--output_dir ./ckpts/ckpt_msvd_retrieval_looseType_tightTransf \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 32 \
--datatype msvd \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32
