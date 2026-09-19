#!/bin/bash
cd /cache/zhangjing/fugue/scion
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
/cache/zhangjing/miniconda3/envs/mm3/bin/python expB_mm3vae.py --a mf-1-000415 > expB_mm3.log 2>&1
echo "exit=$?" >> expB_mm3.log
