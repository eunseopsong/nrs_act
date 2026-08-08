#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 scripts/flow/train_flow_single_cam.py \
  --dataset_dir datasets/polishing/single_cam/20260801_2233/imitation_form_tcp_roi_square10pct_action_fxy_zero \
  --image_backbone dinov3 \
  --dino_roi_pooling attention \
  --freeze_image_backbone
