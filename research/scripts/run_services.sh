#!/bin/bash
# 两个常驻服务同卡:A = SheetSage2 + YuE2(graphtokenizer env,8650),B = adapter v4 + MM3 DiT/vocoder(mm3 env,8651)
cd /cache/zhangjing/fugue/scion
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
GPU=${GPU:-7}; PA=${PA:-8650}; PB=${PB:-8651}
CUDA_VISIBLE_DEVICES=$GPU /cache/zhangjing/miniconda3/envs/graphtokenizer/bin/python fugue_yue2_service.py --port $PA --preload_ss2 > svcA.$GPU.log 2>&1 &
CUDA_VISIBLE_DEVICES=$GPU /cache/zhangjing/miniconda3/envs/mm3/bin/python fugue_graft_service.py --port $PB --run v4 > svcB.$GPU.log 2>&1 &
wait
