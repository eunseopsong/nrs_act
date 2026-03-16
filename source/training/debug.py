#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch

_DIM_NAMES_9 = ["x", "y", "z", "wx", "wy", "wz", "fx", "fy", "fz"]
_FORCE_NAMES_3 = ["fx", "fy", "fz"]
_RGB_NAMES = ["r", "g", "b"]


def _print_1d_mean_std(prefix: str, names, mean_vec: torch.Tensor, std_vec: torch.Tensor):
    mean_vec = mean_vec.detach().cpu().numpy().astype(np.float64)
    std_vec = std_vec.detach().cpu().numpy().astype(np.float64)
    for i, n in enumerate(names):
        print(f"{prefix} {n}: mean={mean_vec[i]:+.6f} std={std_vec[i]:.6f}")


def _unpack_debug_batch(batch):
    """
    Supports both:
      1) image, qpos, action, is_pad
      2) image, qpos, action, is_pad, force_history
    """
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"batch must be tuple/list, got {type(batch)}")

    if len(batch) == 4:
        image, qpos, action, is_pad = batch
        force_history = None
    elif len(batch) == 5:
        image, qpos, action, is_pad, force_history = batch
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")

    return image, qpos, action, is_pad, force_history


@torch.no_grad()
def debug_norm_once(loader, tag: str, max_batches: int = 1):
    for b_idx, batch in enumerate(loader):
        image, qpos, action, is_pad, force_history = _unpack_debug_batch(batch)

        shape_msg = (
            f"\n[NORM-DEBUG/{tag}] shapes: "
            f"image={tuple(image.shape)} qpos={tuple(qpos.shape)} "
            f"action={tuple(action.shape)} is_pad={tuple(is_pad.shape)}"
        )
        if force_history is not None:
            shape_msg += f" force_history={tuple(force_history.shape)}"
        print(shape_msg)

        # --------------------------------------------------
        # qpos stats
        # --------------------------------------------------
        q_mean = qpos.mean(dim=0)
        q_std = qpos.std(dim=0, unbiased=False)
        _print_1d_mean_std(f"[NORM-DEBUG/{tag}] qpos(norm)", _DIM_NAMES_9, q_mean, q_std)

        # --------------------------------------------------
        # action stats (valid only)
        # --------------------------------------------------
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

        # --------------------------------------------------
        # image stats
        # image: (B,K,3,H,W)
        # --------------------------------------------------
        img_ch_mean = image.mean(dim=(0, 1, 3, 4))
        img_ch_std = image.std(dim=(0, 1, 3, 4), unbiased=False)
        _print_1d_mean_std(f"[NORM-DEBUG/{tag}] image([0,1])", _RGB_NAMES, img_ch_mean, img_ch_std)

        # --------------------------------------------------
        # force history stats (if present)
        # force_history: (B,T,3)
        # --------------------------------------------------
        if force_history is not None:
            fh_mean = force_history.mean(dim=(0, 1))
            fh_std = force_history.std(dim=(0, 1), unbiased=False)
            _print_1d_mean_std(
                f"[NORM-DEBUG/{tag}] force_history(norm)",
                _FORCE_NAMES_3,
                fh_mean,
                fh_std,
            )

        # --------------------------------------------------
        # range checks
        # --------------------------------------------------
        q_minv, q_maxv = float(qpos.min().item()), float(qpos.max().item())

        if n_valid > 0:
            a_minv = float(action[valid].min().item())
            a_maxv = float(action[valid].max().item())
        else:
            a_minv, a_maxv = 0.0, 0.0

        im_minv, im_maxv = float(image.min().item()), float(image.max().item())

        range_msg = (
            f"[NORM-DEBUG/{tag}] range check: "
            f"qpos=[{q_minv:.6f},{q_maxv:.6f}] "
            f"action(valid)=[{a_minv:.6f},{a_maxv:.6f}] "
            f"image=[{im_minv:.6f},{im_maxv:.6f}]"
        )

        if force_history is not None:
            fh_minv = float(force_history.min().item())
            fh_maxv = float(force_history.max().item())
            range_msg += f" force_history=[{fh_minv:.6f},{fh_maxv:.6f}]"

        print(range_msg)

        if b_idx + 1 >= max_batches:
            break


def make_grad_scaler(use_amp: bool, device: torch.device):
    enabled = bool(use_amp and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(use_amp: bool, device: torch.device):
    enabled = bool(use_amp and device.type == "cuda")
    try:
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled)