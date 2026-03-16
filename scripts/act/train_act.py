#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ACT training & evaluation script (position+force+image -> position+force)

Normalization:
- qpos/action (9D): per-dimension MIN-MAX -> [0,1]
  x y z wx wy wz fx fy fz  각각 min/max로 정규화
- image: uint8 -> float in [0,1]  (policy.py 내부에서 ImageNet normalize는 그대로 유지)

IMPORTANT:
- policy.py 유지
- 데이터 로딩(episode_*.hdf5 검색) 유지
- 멀티카메라 입력 형태 (B,K,3,H,W) 유지
- debug_norm: 학습 시작 시 "정규화 이후" 9D + RGB(3ch) 각각 mean/std 출력
"""

import os
import sys
import pickle
import argparse
from datetime import datetime

# -------------------------------------------------------------------------
# source/ 경로 추가
# scripts/act/train_act.py 에서 source/*.py 를 import 가능하게 함
# -------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SOURCE_DIR = os.path.join(_PROJECT_ROOT, "source")

for p in [_PROJECT_ROOT, _SOURCE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from utils import load_data
from act_train_utils import train_bc, find_latest_timestamped_subdir, make_policy


# -----------------------------
# Robust TASK_CONFIGS import (optional)
# -----------------------------
TASK_CONFIGS = {}
try:
    from custom.custom_constants import TASK_CONFIGS as _TC  # optional
    TASK_CONFIGS = _TC
except Exception:
    TASK_CONFIGS = {}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] using device = {device}")

    is_eval = args.eval
    task_name = args.task_name
    ckpt_root_dir = args.ckpt_dir
    policy_class = args.policy_class
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    seed = args.seed
    lr = args.lr

    dataset_dir = args.dataset_dir
    num_episodes = args.num_episodes

    if dataset_dir is None:
        if task_name in TASK_CONFIGS and "dataset_dir" in TASK_CONFIGS[task_name]:
            dataset_dir = TASK_CONFIGS[task_name]["dataset_dir"]
        else:
            raise ValueError("dataset_dir is not provided and TASK_CONFIGS has no entry for this task.")

    if num_episodes <= 0:
        if task_name in TASK_CONFIGS and "num_episodes" in TASK_CONFIGS[task_name]:
            num_episodes = int(TASK_CONFIGS[task_name]["num_episodes"])
        else:
            num_episodes = 0  # load_data will use all episodes

    # cameras (KEEP)
    camera_names = ["cam_top", "cam_ee"]

    print(f"[INFO] task_name      = {task_name}")
    print(f"[INFO] dataset_dir    = {dataset_dir}")
    print(f"[INFO] num_episodes   = {num_episodes if num_episodes > 0 else 'ALL'}")
    print(f"[INFO] camera_names   = {camera_names}")
    print(f"[INFO] chunk_size     = {args.chunk_size}")
    print(f"[INFO] train_seq_len  = {args.train_seq_len}")
    print(f"[INFO] val_seq_len    = {args.val_seq_len}")
    print(f"[INFO] samples/ep     = {args.samples_per_episode}")
    print(f"[INFO] AMP            = {args.amp}")
    print(f"[INFO] NORM           = min-max per-dim -> [0,1] (qpos/action), image -> [0,1]")

    # policy config
    lr_backbone = args.lr_backbone
    backbone = args.backbone

    if policy_class == "ACT":
        policy_config = {
            "lr": lr,
            "num_queries": args.chunk_size,
            "kl_weight": args.kl_weight,
            "hidden_dim": args.hidden_dim,
            "dim_feedforward": args.dim_feedforward,
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": args.enc_layers,
            "dec_layers": args.dec_layers,
            "nheads": args.nheads,
            "camera_names": camera_names,
            "state_dim": 9,
            "action_dim": 9,
            "image_resize_hw": args.image_resize_hw,
            "image_pool_hw": args.image_pool_hw,
            "pretrained_backbone": (not args.no_pretrained),
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": lr,
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
            "state_dim": 9,
            "action_dim": 9,
            "image_resize_hw": args.image_resize_hw,
            "image_pool_hw": args.image_pool_hw,
            "pretrained_backbone": (not args.no_pretrained),
        }
    else:
        raise NotImplementedError(policy_class)

    # ============================================================
    # [EVAL MODE]
    # ============================================================
    if is_eval:
        ckpt_dir = ckpt_root_dir
        best_ckpt = os.path.join(ckpt_dir, "policy_best.ckpt")
        if not os.path.exists(best_ckpt):
            latest_sub = find_latest_timestamped_subdir(ckpt_root_dir)
            if latest_sub is None:
                raise FileNotFoundError(
                    f"[EVAL] No policy_best.ckpt in {ckpt_root_dir} "
                    f"and no timestamped subdirectories were found."
                )
            ckpt_dir = latest_sub
            best_ckpt = os.path.join(ckpt_dir, "policy_best.ckpt")

        if not os.path.exists(best_ckpt):
            raise FileNotFoundError(f"[EVAL] policy_best.ckpt not found in {ckpt_dir}")

        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"[EVAL] dataset_stats.pkl not found in {ckpt_dir}")

        print(f"[EVAL] Using checkpoint dir: {ckpt_dir}")
        print(f"[INFO] Loading checkpoint from {best_ckpt}")

        policy = make_policy(policy_class, policy_config).to(device)

        ckpt = torch.load(best_ckpt, map_location=device)
        if policy_class == "ACT":
            policy.model.load_state_dict(ckpt, strict=False)
        else:
            policy.load_state_dict(ckpt, strict=False)

        policy.eval()

        with open(stats_path, "rb") as f:
            _ = pickle.load(f)
        print(f"[INFO] Loaded dataset stats from {stats_path}")

        print("\n✅ Model ready for inference!")
        print("   (주의: action은 [0,1] 정규화 스케일이므로, 실제 제어 시 stats로 denormalize 필요)\n")
        return

    # ============================================================
    # [TRAIN MODE]
    # ============================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_dir = os.path.join(ckpt_root_dir, timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[TRAIN] Checkpoints will be saved under: {ckpt_dir}")

    train_loader, val_loader, stats, meta = load_data(
        dataset_dir=dataset_dir,
        num_episodes=num_episodes,
        camera_names=camera_names,
        batch_size_train=batch_size,
        batch_size_val=batch_size,
        seq_len_train=args.train_seq_len,
        seq_len_val=args.val_seq_len,
        seed=seed,
        samples_per_episode=args.samples_per_episode,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    print(f"[INFO] data meta: {meta}")

    with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)
    print(f"[INFO] saved dataset stats -> {ckpt_dir}/dataset_stats.pkl")

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "policy_class": policy_class,
        "policy_config": policy_config,
        "seed": seed,
        "device": device,
        "amp": args.amp,
        "debug_norm": args.debug_norm,
        "debug_norm_batches": 1,
    }

    best_ckpt_info = train_bc(train_loader, val_loader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    best_ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    torch.save(best_state_dict, best_ckpt_path)
    print(f"[INFO] Best ckpt saved -> {best_ckpt_path} (val_loss={min_val_loss:.6f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--eval", action="store_true", help="run inference instead of training")
    p.add_argument("--ckpt_dir", type=str, required=True, help="root checkpoint directory")
    p.add_argument("--policy_class", type=str, required=True, choices=["ACT", "CNNMLP"])
    p.add_argument("--task_name", type=str, required=True)

    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num_epochs", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)

    # dataset override
    p.add_argument("--dataset_dir", type=str, default=None)
    p.add_argument("--num_episodes", type=int, default=0)

    # IMPORTANT: sequence length
    p.add_argument("--chunk_size", type=int, default=100)
    p.add_argument("--train_seq_len", type=int, default=100)
    p.add_argument("--val_seq_len", type=int, default=100)

    # IMPORTANT: restore epoch steps
    p.add_argument("--samples_per_episode", type=int, default=50)

    # ACT params
    p.add_argument("--kl_weight", type=float, default=10)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)

    # transformer depth
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--enc_layers", type=int, default=4)
    p.add_argument("--dec_layers", type=int, default=7)

    # backbone
    p.add_argument("--backbone", type=str, default="resnet18")
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--no_pretrained", action="store_true", default=False)

    # image perf knobs
    p.add_argument("--image_resize_hw", type=int, default=256)
    p.add_argument("--image_pool_hw", type=int, default=4)

    # dataloader knobs
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=2)

    # amp
    p.add_argument("--amp", action="store_true", default=False)

    # debug normalization (print-only)
    p.add_argument(
        "--debug_norm",
        action="store_true",
        help="print post-normalization per-dim mean/std (9D + RGB), then continue"
    )

    args = p.parse_args()

    # CRITICAL FIX:
    # Some DETR/ACT modules parse sys.argv again internally.
    # Remove our custom flag to prevent "unrecognized arguments" crash.
    if getattr(args, "debug_norm", False):
        sys.argv = [a for a in sys.argv if a != "--debug_norm"]

    main(args)