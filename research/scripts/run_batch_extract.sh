#!/bin/bash
# 两侧整库提取:GPU0 YuE2 VAE latent(全部 split),GPU6 MM3 TF c25(val→test→train)
cd /cache/zhangjing/fugue/scion
export HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 /cache/zhangjing/miniconda3/envs/graphtokenizer/bin/python yue2_encode.py > yue2_encode.log 2>&1 &
(
  for s in val test train; do
    CUDA_VISIBLE_DEVICES=6 /cache/zhangjing/miniconda3/envs/mm3/bin/python mm3_tf.py --split $s >> tf_c25.log 2>&1
  done
  echo "TF ALL DONE exit=$?" >> tf_c25.log
) &
wait
