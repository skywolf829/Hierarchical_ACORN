#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/Hierarchical_ACORN

python3 Code/train.py --save_name Cat_16GPUs --use_residual false --log_img true --train_distributed true --gpus_per_node 8 --num_nodes 2