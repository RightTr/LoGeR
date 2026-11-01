#!/bin/bash

# Usage: bash demo_run.sh [CUDA_DEVICE] [INPUT_PATH]
# Example: bash demo_run.sh 0 /path/to/your/data

# Get arguments
CUDA_DEVICE=$1
INPUT_PATH=$2

if [ -n "$CUDA_DEVICE" ]; then
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
fi

# choose from LoGeR, LoGeR_star
ckpt_list=( 
    "LoGeR"
    "LoGeR_star"
)

for ckpt_name in "${ckpt_list[@]}"; do
    echo "--- Processing checkpoint: $ckpt_name ---"
    config_path="ckpts/${ckpt_name}/original_config.yaml"
    model_path="ckpts/${ckpt_name}/latest.pt"

    input_path="${INPUT_PATH:-data/examples/office}"

    echo "Running evaluation..."

    python  demo_viser.py \
        --input "$input_path" \
        --config "$config_path" \
        --model_name "$model_path" \
        --start_frame 0 \
        --end_frame 10000 \
        --stride 40 \
        --window_size 32 \
        --overlap_size 3 \
        --subsample 2 \
        --share \
        --port 8081 \
        # --reset_every 5  # turned on for extreme long sequences (>1k frames)


    echo "--- Finished processing $ckpt_name ---"
    echo ""
done

echo "All checkpoints have been processed."