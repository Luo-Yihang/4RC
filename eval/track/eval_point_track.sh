#!/bin/bash
set -e

# Fix libstdc++ compatibility issue
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Point-based (Sparse) Tracking Evaluation Script
# Evaluates on datasets with sparse point annotations

workdir='.'
checkpoint="Luo-Yihang/4RC"
model_type="4rc"

# Sparse tracking datasets
datasets=('pstudio_mini' 'po_mini' 'adt_mini' 'ds_mini')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/point_track/${data}"
    echo "Evaluating point tracking on ${data}..."
    echo "Output dir: $output_dir"

    python eval/track/launch.py \
        --checkpoint_dir="$checkpoint" \
        --output_dir="$output_dir" \
        --data_type="$data" \
        --data_root="./data" \
        --num_frames=64 \
        --pose_eval_stride=1 \
        --size=512 \
        --patch_size=14 \
        --model_type="$model_type" \
        --align="sim3" \
        --save_predictions \
        --seed=42

    echo "✓ Completed $data"
    echo "---"
done

echo "✅ Point tracking evaluation completed!"
echo "Results saved in: eval_results/point_track/"
