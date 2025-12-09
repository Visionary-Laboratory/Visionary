#!/bin/bash

REF_IMG_DIR=example_imgs
OUT_Root=outputs
MOTION_NAME="CMU/CMU/06/06_01_poses" # you can change motion sequences

mkdir -p "$OUT_Root/gaussians"
mkdir -p "$OUT_Root/smplx_json/$MOTION_NAME"
mkdir -p "$OUT_Root/onnx/$MOTION_NAME"

echo "--- Converting Motion Data (Once) ---"

python convert_cmu.py \
  --npz_path motions/${MOTION_NAME}.npz \
  --output_director $OUT_Root/smplx_json/$MOTION_NAME \
  --max_frame_num 1000 \
  --frame_stride 1


for img_path in "$REF_IMG_DIR"/*; do
    

    filename=$(basename -- "$img_path")
    name="${filename%.*}"

    echo "============================================"
    echo "Processing Character: $name"
    echo "============================================"

    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python run_lhm.py \
      --character_image_path "$img_path" \
      --save_gaussian_path "$OUT_Root/gaussians/$name.pth"

    python onnx_template_lhm.py \
      --pth "$OUT_Root/gaussians/$name.pth" \
      --motion_json "$OUT_Root/smplx_json/$MOTION_NAME" \
      --out "$OUT_Root/onnx/$MOTION_NAME/$name.onnx"

    python kill_big_cat.py \
      --input "$OUT_Root/onnx/$MOTION_NAME/$name.onnx" \
      --output "$OUT_Root/onnx/$MOTION_NAME/$name.onnx"

    echo "Finished processing: $name"
    
done

echo "All done!"