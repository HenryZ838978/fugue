#!/bin/bash
# 等两侧整库提取结束,再在全量 train 上训 v2(8 层 d768,crop 768,20k 步)
cd /cache/zhangjing/fugue/scion
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7
while ! grep -q "TF ALL DONE" tf_c25.log; do sleep 30; done
while ! grep -q "^DONE" yue2_encode.log; do sleep 30; done
/cache/zhangjing/miniconda3/envs/mm3/bin/python train_adapter.py --name v2 --layers 8 --d 768 --heads 12 --crop 768 --bs 24 --steps 20000 --eval_every 2000 --n_val 150 --workers 10 > runs_v2.log 2>&1
echo "exit=$?" >> runs_v2.log
