#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/Hierarchical_ACORN

#python3 Code/train.py --save_name Cat_16GPUs --use_residual false --log_img true --train_distributed true --gpus_per_node 8 --num_nodes 2

nohup python3 -u Code/train.py --use_residual true --FC_size_exp_start 1.0 --FC_size_exp_grow 1.5 \
--octree_depth_end 7 --save_name Isomag3D_8gpus_residual --target_signal TrainingData/isomag3D.h5 \
--num_channels 1 --mode 3D --local_queries_per_iter 1000000 --train_distributed true --num_nodes 1 \
--gpus_per_node 8 
