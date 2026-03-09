#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT training & evaluation script (position+force+image -> position+force)

Normalization (UPDATED per your request):
- qpos/action (9D): per-dimension MIN-MAX -> [0,1]
  x y z wx wy wz fx fy fz  각각 min/max로 정규화
- image: uint8 -> float in [0,1]  (policy.py 내부에서 ImageNet normalize는 그대로 유지)

IMPORTANT:
- policy.py 유지
- 데이터 로딩(episode_*.hdf5 검색) 유지
- 멀티카메라 입력 형태 (B,K,3,H,W) 유지  (여기 깨지면 conv2d 채널 mismatch 터짐)
- debug_norm: 학습 시작 시 "정규화 이후" 9D + RGB(3ch) 각각 mean/std 출력
"""

import os
import sys
import pickle
import argparse
from datetime import datetime
from copy import deepcopy
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import (
    load_data,
    compute_dict_mean,
    set_seed,
)

from policy import ACTPolicy, CNNMLPPolicy


# -----------------------------
# Robust TASK_CONFIGS import (optional)
# -----------------------------
TASK_CONFIGS = {}
try:
    from custom.custom_constants import TASK_CONFIGS as _TC  # optional
    TASK_CONFIGS = _TC
except Exception:
    TASK_CONFIGS = {}


# -------------------------------------------------------------------------
# 유틸: 루트 디렉터리 아래에서 타임스탬프 폴더 중 최신 찾기
# -------------------------------------------------------------------------
def find_latest_timestamped_subdir(root_dir: str) -> Optional[str]:
    if not os.path.isdir(root_dir):
        return None

    candidates = []
    for name in os.listdir(root_dir):
        sub = os.path.join(root_dir, name)
        if not os.path.isdir(sub):
            continue

        ok = False
        for fmt in ("%Y%m%d_%H%M", "%m%d_%H%M"):
            try:
                datetime.strptime(name, fmt)
                ok = True
                break
            except ValueError:
                pass

        if ok:
            candidates.append((name, sub))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# -------------------------------------------------------------------------
# HELPER
# -------------------------------------------------------------------------
def make_policy(policy_class: str, policy_config: dict):
    if policy_class == "ACT":
        return ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        return CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError(policy_class)


def forward_pass(batch, policy, device):
    """
    batch: (image, qpos, action, is_pad)
      image : (B, K, 3, H, W) float [0,1]
      qpos  : (B, 9)          normalized to [0,1]
      action: (B, T, 9)       normalized to [0,1]
      is_pad: (B, T)          bool
    """
    image, qpos, action, is_pad = batch
    image = image.to(device, non_blocking=True)
    qpos = qpos.to(device, non_blocking=True)
    action = action.to(device, non_blocking=True)
    is_pad = is_pad.to(device, non_blocking=True)
    return policy(qpos, image, action, is_pad)


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    if len(train_history) == 0 or len(validation_history) == 0:
        return

    keys = list(train_history[0].keys())
    for key in keys:
        plt.figure()
        train_values = [float(x[key]) for x in train_history if key in x]
        val_values = [float(x[key]) for x in validation_history if key in x]

        plt.plot(np.linspace(0, num_epochs - 1, len(train_values)), train_values, label="train")
        plt.plot(np.linspace(0, num_epochs - 1, len(val_values)), val_values, label="val")

        plt.legend()
        plt.tight_layout()
        plt.title(key)
        plt.savefig(os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png"))
        plt.close()

    print(f"[INFO] Saved plots to {ckpt_dir}")


# -------------------------------------------------------------------------
# Debug norm: per-dim mean/std (9D + RGB)
# -------------------------------------------------------------------------
_DIM_NAMES_9 = ["x", "y", "z", "wx", "wy", "wz", "fx", "fy", "fz"]
_RGB_NAMES = ["r", "g", "b"]


def _print_1d_mean_std(prefix: str, names, mean_vec: torch.Tensor, std_vec: torch.Tensor):
    mean_vec = mean_vec.detach().cpu().numpy().astype(np.float64)
    std_vec = std_vec.detach().cpu().numpy().astype(np.float64)
    for i, n in enumerate(names):
        print(f"{prefix} {n}: mean={mean_vec[i]:+.6f} std={std_vec[i]:.6f}")


@torch.no_grad()
def debug_norm_once(loader, tag: str, max_batches: int = 1):
    """
    Prints post-normalization mean/std for:
      - qpos (9 dims)
      - action (9 dims, valid only)
      - image RGB channels (3 dims)  (image is [0,1] before policy.py normalize)
    """
    for b_idx, batch in enumerate(loader):
        image, qpos, action, is_pad = batch
        # image: (B,K,3,H,W), qpos:(B,9), action:(B,T,9), is_pad:(B,T)
        print(f"\n[NORM-DEBUG/{tag}] shapes: image={tuple(image.shape)} qpos={tuple(qpos.shape)} action={tuple(action.shape)} is_pad={tuple(is_pad.shape)}")

        # ---- qpos (normalized) per-dim ----
        q_mean = qpos.mean(dim=0)
        q_std = qpos.std(dim=0, unbiased=False)
        _print_1d_mean_std(f"[NORM-DEBUG/{tag}] qpos(norm)", _DIM_NAMES_9, q_mean, q_std)

        # ---- action (normalized) per-dim (valid only) ----
        valid = ~is_pad  # (B,T)
        n_valid = int(valid.sum().item())
        print(f"[NORM-DEBUG/{tag}] valid action steps: {n_valid} / {int(is_pad.numel())}")
        if n_valid > 0:
            a_valid = action[valid]  # (n_valid, 9)
            a_mean = a_valid.mean(dim=0)
            a_std = a_valid.std(dim=0, unbiased=False)
            _print_1d_mean_std(f"[NORM-DEBUG/{tag}] action(norm)", _DIM_NAMES_9, a_mean, a_std)
        else:
            print(f"[NORM-DEBUG/{tag}] ⚠️ no valid timesteps (check is_pad/valid_len)")

        # ---- image [0,1] RGB channel stats ----
        # image mean/std over (B,K,H,W) for each channel
        # image: (B,K,3,H,W)
        img_ch_mean = image.mean(dim=(0, 1, 3, 4))  # (3,)
        img_ch_std = image.std(dim=(0, 1, 3, 4), unbiased=False)  # (3,)
        _print_1d_mean_std(f"[NORM-DEBUG/{tag}] image([0,1])", _RGB_NAMES, img_ch_mean, img_ch_std)

        # optional quick range sanity
        q_minv, q_maxv = float(qpos.min().item()), float(qpos.max().item())
        a_minv, a_maxv = float(action[valid].min().item()) if n_valid > 0 else 0.0, float(action[valid].max().item()) if n_valid > 0 else 0.0
        im_minv, im_maxv = float(image.min().item()), float(image.max().item())
        print(f"[NORM-DEBUG/{tag}] range check: qpos=[{q_minv:.6f},{q_maxv:.6f}] action(valid)=[{a_minv:.6f},{a_maxv:.6f}] image=[{im_minv:.6f},{im_maxv:.6f}]")

        if b_idx + 1 >= max_batches:
            break


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    device = config["device"]
    use_amp = config["amp"]
    debug_norm = config.get("debug_norm", False)
    debug_norm_batches = int(config.get("debug_norm_batches", 1))

    set_seed(seed)

    # perf flags (optional; keep)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # debug norm at start (requested)
    if debug_norm:
        print("[INFO] debug_norm enabled: printing post-normalization per-dim mean/std for TRAIN and VAL (then continue).")
        try:
            debug_norm_once(train_dataloader, tag="TRAIN", max_batches=debug_norm_batches)
        except Exception as e:
            print(f"[WARN] NORM-DEBUG/TRAIN failed: {e}")
        try:
            debug_norm_once(val_dataloader, tag="VAL", max_batches=debug_norm_batches)
        except Exception as e:
            print(f"[WARN] NORM-DEBUG/VAL failed: {e}")

    policy = make_policy(policy_class, policy_config).to(device)
    optimizer = policy.configure_optimizers()

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    train_history, validation_history = [], []
    min_val_loss = float("inf")
    best_ckpt_info = None

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[DEBUG] Policy class = {policy_class}, trainable params = {n_params/1e6:.2f}M")

    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch {epoch}")

        # ---------------- Validation ----------------
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch in val_dataloader:
                with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                    out = forward_pass(batch, policy, device)
                epoch_dicts.append({k: out[k].detach().float().cpu() for k in out})

            epoch_summary = compute_dict_mean(epoch_dicts)
            epoch_summary = {k: float(epoch_summary[k]) for k in epoch_summary}
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            print("Val:", " | ".join([f"{k}:{epoch_summary[k]:.6f}" for k in epoch_summary]))

            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                if policy_class == "ACT":
                    best_state = deepcopy(policy.model.state_dict())
                else:
                    best_state = deepcopy(policy.state_dict())
                best_ckpt_info = (epoch, min_val_loss, best_state)

        # ---------------- Training ----------------
        policy.train()
        optimizer.zero_grad(set_to_none=True)

        batch_dicts = []
        for batch_idx, batch in enumerate(train_dataloader):
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                forward_dict = forward_pass(batch, policy, device)
                loss = forward_dict["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            batch_dicts.append({k: forward_dict[k].detach().float().cpu() for k in forward_dict})

            if epoch == 0 and batch_idx < 3:
                print(f"[DEBUG] Epoch 0, batch {batch_idx}, train loss = {float(loss):.6f}")

        epoch_train_summary = compute_dict_mean(batch_dicts)
        epoch_train_summary = {k: float(epoch_train_summary[k]) for k in epoch_train_summary}
        train_history.append(epoch_train_summary)
        print("Train:", " | ".join([f"{k}:{epoch_train_summary[k]:.6f}" for k in epoch_train_summary]))

        # ---------------- Save intermediate ckpt ----------------
        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            if policy_class == "ACT":
                state_to_save = policy.model.state_dict()
            else:
                state_to_save = policy.state_dict()

            torch.save(state_to_save, ckpt_path)
            print(f"[INFO] Saved intermediate ckpt -> {ckpt_path}")
            plot_history(train_history, validation_history, epoch + 1, ckpt_dir, seed)

    best_epoch, best_loss, best_state_dict = best_ckpt_info
    print(f"[INFO] Best epoch = {best_epoch}, min val loss = {best_loss:.6f}")
    return best_ckpt_info


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
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

    # policy config (KEEP existing)
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


# -------------------------------------------------------------------------
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

    # transformer depth (old ACT default: enc=4, dec=7)
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
    p.add_argument("--debug_norm", action="store_true",
                   help="print post-normalization per-dim mean/std (9D + RGB), then continue")

    args = p.parse_args()

    # CRITICAL FIX:
    # Some DETR/ACT modules parse sys.argv again internally.
    # Remove our custom flag to prevent "unrecognized arguments" crash.
    if getattr(args, "debug_norm", False):
        sys.argv = [a for a in sys.argv if a != "--debug_norm"]

    main(args)