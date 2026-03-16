#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List

import h5py
import numpy as np


def compute_norm_stats_all(episode_files: List[str]) -> Dict[str, np.ndarray]:
    q_min = q_max = None
    a_min = a_max = None

    for p in episode_files:
        with h5py.File(p, "r") as h:
            pos = np.asarray(h["/observations/position"][()], dtype=np.float32)  # (T,6)
            frc = np.asarray(h["/observations/force"][()], dtype=np.float32)     # (T,3)
            q = np.concatenate([pos, frc], axis=-1)                              # (T,9)

            a_pos = np.asarray(h["/action/position"][()], dtype=np.float32)      # (T,6)
            a_frc = np.asarray(h["/action/force"][()], dtype=np.float32)         # (T,3)
            a = np.concatenate([a_pos, a_frc], axis=-1)                          # (T,9)

            is_pad = np.asarray(h["/observations/is_pad"][()], dtype=np.bool_)
            valid = ~is_pad
            if valid.sum() == 0:
                continue

            qv = q[valid]
            av = a[valid]

            qv_min = np.min(qv, axis=0)
            qv_max = np.max(qv, axis=0)
            av_min = np.min(av, axis=0)
            av_max = np.max(av, axis=0)

            if q_min is None:
                q_min = qv_min
                q_max = qv_max
                a_min = av_min
                a_max = av_max
            else:
                q_min = np.minimum(q_min, qv_min)
                q_max = np.maximum(q_max, qv_max)
                a_min = np.minimum(a_min, av_min)
                a_max = np.maximum(a_max, av_max)

    if q_min is None:
        q_min = np.zeros((9,), dtype=np.float32)
        q_max = np.ones((9,), dtype=np.float32)
        a_min = np.zeros((9,), dtype=np.float32)
        a_max = np.ones((9,), dtype=np.float32)

    eps = 1e-6
    q_rng = np.maximum(q_max - q_min, eps)
    a_rng = np.maximum(a_max - a_min, eps)
    q_max = q_min + q_rng
    a_max = a_min + a_rng

    return {
        "qpos_min": q_min.astype(np.float32),
        "qpos_max": q_max.astype(np.float32),
        "action_min": a_min.astype(np.float32),
        "action_max": a_max.astype(np.float32),
    }


def denormalize_action(action_norm: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    if "action_min" in stats and "action_max" in stats:
        mn = stats["action_min"].reshape((1,) * (action_norm.ndim - 1) + (9,))
        mx = stats["action_max"].reshape((1,) * (action_norm.ndim - 1) + (9,))
        return action_norm * (mx - mn) + mn

    mu = stats["action_mean"].reshape((1,) * (action_norm.ndim - 1) + (9,))
    sd = stats["action_std"].reshape((1,) * (action_norm.ndim - 1) + (9,))
    return action_norm * sd + mus