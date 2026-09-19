#!/bin/bash
cd /cache/zhangjing/fugue/scion
/cache/zhangjing/miniconda3/envs/graphtokenizer/bin/python fugue_arena.py --port 7877 > arena.log 2>&1
