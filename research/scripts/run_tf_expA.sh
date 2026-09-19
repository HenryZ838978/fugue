#!/bin/bash
cd /cache/zhangjing/fugue/scion
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6
/cache/zhangjing/miniconda3/envs/mm3/bin/python mm3_tf.py --ids mf-1-000415 mf-1-001136 --save_hidden > tf_expA.log 2>&1
echo "exit=$?" >> tf_expA.log
