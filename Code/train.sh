#!/bin/sh
cd /lus/theta-fs0/projects/DL4VIS/Hierarchical_ACORN

echo ${PBS_ARRAYID}
echo $PBS_ARRAY_INDEX
#python3 Code/main.py --save_name Temp --node_num $PBS_ARRAYID