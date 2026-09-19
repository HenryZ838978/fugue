#!/bin/bash
cd /cache/zhangjing/fugue/scion
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7
/cache/zhangjing/miniconda3/envs/mm3/bin/python probe_linear.py > probe_linear.log 2>&1
echo "exit=$?" >> probe_linear.log
