#!/bin/bash
cd /cache/zhangjing/fugue/scion
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7
/cache/zhangjing/miniconda3/envs/mm3/bin/python expA.py --a mf-1-000415 --b mf-1-001136 > expA.log 2>&1
echo "exit=$?" >> expA.log
