#!/bin/bash

AVATAR_PATH=demo/avatar.pth
MOTION_PATH=demo/motion
OUT_ONNX_PATH=out/out.onnx

python onnx_template_lhm.py \
  --pth $AVATAR_PATH \
  --motion_json $MOTION_PATH \
  --out $OUT_ONNX_PATH

python kill_big_cat.py \
  --input $OUT_ONNX_PATH \
  --output $OUT_ONNX_PATH
