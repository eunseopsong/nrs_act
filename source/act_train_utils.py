#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training helper utilities for ACT/CNNMLP training.
"""

import os
from datetime import datetime
from copy import deepcopy
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import compute_dict_mean, set_seed
from models.policy import ACTPolicy, CNNMLPPolicy


# -------------------------------------------------------------------------
# Filesystem helper
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
# Policy helper
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
      qpos  : (B, 9)
      action: (B, T, 9)
      is_pad: (B, T)
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
# Debug norm
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
    for b_idx, batch in enumerate(loader):
        image, qpos, action, is_pad = batch
        print(
            f"\n[NORM-DEBUG/{tag}] shapes: "
            f"image={tuple(image.shape)} qpos={tuple(qpos.shape)} "
            f"action={tuple(action.shape)} is_pad={tuple(is_pad.shape)}"
        )

        q_mean = qpos.mean(dim=0)
        q_std = qpos.std(dim=0, unbiased=False)
        _print_1d_mean_std(f"[NORM-DEBUG/{tag}] qpos(norm)", _DIM_NAMES_9, q_mean, q_std)

        valid = ~is_pad
        n_valid = int(valid.sum().item())
        print(f"[NORM-DEBUG/{tag}] valid action steps: {n_valid} / {int(is_pad.numel())}")
        if n_valid > 0:
            a_valid = action[valid]
            a_mean = a_valid.mean(dim=0)
            a_std = a_valid.std(dim=0, unbiased=False)
            _print_1d_mean_std(f"[NORM-DEBUG/{tag}] action(norm)", _DIM_NAMES_9, a_mean, a_std)
        else:
            print(f"[NORM-DEBUG/{tag}] ⚠️ no valid timesteps")

        img_ch_mean = image.mean(dim=(0, 1, 3, 4))
        img_ch_std = image.std(dim=(0, 1, 3, 4), unbiased=False)
        _print_1d_mean_std(f"[NORM-DEBUG/{tag}] image([0,1])", _RGB_NAMES, img_ch_mean, img_ch_std)

        q_minv, q_maxv = float(qpos.min().item()), float(qpos.max().item())
        if n_valid > 0:
            a_minv = float(action[valid].min().item())
            a_maxv = float(action[valid].max().item())
        else:
            a_minv, a_maxv = 0.0, 0.0
        im_minv, im_maxv = float(image.min().item()), float(image.max().item())

        print(
            f"[NORM-DEBUG/{tag}] range check: "
            f"qpos=[{q_minv:.6f},{q_maxv:.6f}] "
            f"action(valid)=[{a_minv:.6f},{a_maxv:.6f}] "
            f"image=[{im_minv:.6f},{im_maxv:.6f}]"
        )

        if b_idx + 1 >= max_batches:
            break


# -------------------------------------------------------------------------
# AMP helpers
# -------------------------------------------------------------------------
def _make_grad_scaler(use_amp: bool, device: torch.device):
    enabled = bool(use_amp and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(use_amp: bool, device: torch.device):
    enabled = bool(use_amp and device.type == "cuda")
    try:
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled)


# -------------------------------------------------------------------------
# Train
# -------------------------------------------------------------------------
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

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if debug_norm:
        print("[INFO] debug_norm enabled: printing post-normalization stats for TRAIN and VAL.")
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
    scaler = _make_grad_scaler(use_amp, device)

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
                with _autocast_context(use_amp, device):
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
            with _autocast_context(use_amp, device):
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