#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _get_valid_len(is_pad: np.ndarray) -> int:
    idx = np.where(is_pad.astype(bool))[0]
    return int(idx[0]) if len(idx) > 0 else int(is_pad.shape[0])


def _resolve_cam_key(img_grp: h5py.Group, requested: str) -> str:
    if requested in img_grp:
        return requested

    fallback = {"cam_top": "cam_front", "cam_ee": "cam_head"}
    alt = fallback.get(requested, None)
    if alt is not None and alt in img_grp:
        return alt

    keys = list(img_grp.keys())
    if len(keys) == 0:
        raise KeyError("No image datasets under /observations/images")
    return keys[0]


def _build_force_history(
    force_full: np.ndarray,
    current_ts: int,
    history_len: int,
) -> np.ndarray:
    """
    force_full : (T, 3)
    current_ts : current observation index t
    history_len: desired history length L

    returns:
        force_history : (L, 3)
        history = [f_{t-L+1}, ..., f_t]
        if insufficient at episode start, left-pad by repeating first available force
    """
    if force_full.ndim != 2 or force_full.shape[1] != 3:
        raise ValueError(f"force_full must have shape (T, 3), got {force_full.shape}")

    T = force_full.shape[0]
    if T <= 0:
        raise ValueError("force_full is empty")

    current_ts = int(np.clip(current_ts, 0, T - 1))
    history_len = int(max(1, history_len))

    hist_start = max(0, current_ts - history_len + 1)
    hist = force_full[hist_start : current_ts + 1]  # (<=L, 3)

    if hist.shape[0] < history_len:
        pad_count = history_len - hist.shape[0]
        pad_value = hist[0:1] if hist.shape[0] > 0 else force_full[0:1]
        pad = np.repeat(pad_value, pad_count, axis=0)
        hist = np.concatenate([pad, hist], axis=0)

    return hist.astype(np.float32)


class EpisodicStartDataset(Dataset):
    """
    Default return (backward-compatible):
      image : (K,3,H,W) float [0,1]
      qpos  : (9,)
      action: (T,9)
      is_pad: (T,)

    If return_force_history=True:
      image         : (K,3,H,W) float [0,1]
      qpos          : (9,)
      action        : (T,9)
      is_pad        : (T,)
      force_history : (L,3)   normalized with qpos force stats
    """

    def __init__(
        self,
        episode_files: List[str],
        camera_names: List[str],
        norm_stats: Dict[str, np.ndarray],
        seq_len: int,
        samples_per_episode: int = 50,
        seed: int = 0,
        is_sim: bool = False,
        force_history_len: int = 10,
        return_force_history: bool = False,
        position_dim: int = 6,
        force_dim: int = 3,
    ):
        self.files = list(episode_files)
        self.camera_names = list(camera_names)
        self.seq_len = int(seq_len)
        self.samples_per_episode = int(max(1, samples_per_episode))
        self.seed = int(seed)
        self.is_sim = bool(is_sim)

        self.force_history_len = int(max(1, force_history_len))
        self.return_force_history = bool(return_force_history)

        self.position_dim = int(position_dim)
        self.force_dim = int(force_dim)

        self.q_min = torch.tensor(norm_stats["qpos_min"], dtype=torch.float32)
        self.q_max = torch.tensor(norm_stats["qpos_max"], dtype=torch.float32)
        self.a_min = torch.tensor(norm_stats["action_min"], dtype=torch.float32)
        self.a_max = torch.tensor(norm_stats["action_max"], dtype=torch.float32)

        eps = 1e-6
        self.q_den = torch.clamp(self.q_max - self.q_min, min=eps)
        self.a_den = torch.clamp(self.a_max - self.a_min, min=eps)

        # qpos = [position(6), force(3)] 기준
        force_start = self.position_dim
        force_end = self.position_dim + self.force_dim
        self.f_min = self.q_min[force_start:force_end]
        self.f_den = self.q_den[force_start:force_end]

        self._rng = np.random.RandomState(self.seed)
        self._worker_rng = {}

    def __len__(self):
        return len(self.files) * self.samples_per_episode

    def _get_rng(self):
        wi = torch.utils.data.get_worker_info()
        if wi is None:
            return self._rng

        wid = wi.id
        if wid not in self._worker_rng:
            self._worker_rng[wid] = np.random.RandomState(self.seed + 10007 * (wid + 1))
        return self._worker_rng[wid]

    def __getitem__(self, idx):
        rng = self._get_rng()
        file_idx = idx % len(self.files)
        path = self.files[file_idx]

        with h5py.File(path, "r") as h:
            is_pad_full = np.asarray(h["/observations/is_pad"][()], dtype=np.bool_)
            valid_len = _get_valid_len(is_pad_full)
            if valid_len <= 0:
                valid_len = int(is_pad_full.shape[0])

            start_ts = int(rng.randint(0, max(1, valid_len)))
            a_start = start_ts if self.is_sim else max(0, start_ts - 1)

            # current observation
            pos = np.asarray(h["/observations/position"][start_ts], dtype=np.float32)
            frc = np.asarray(h["/observations/force"][start_ts], dtype=np.float32)
            qpos = np.concatenate([pos, frc], axis=-1).astype(np.float32)

            # image
            img_grp = h["/observations/images"]
            cam_imgs = []
            for cam in self.camera_names:
                cam_key = _resolve_cam_key(img_grp, cam)
                img = np.asarray(img_grp[cam_key][start_ts], dtype=np.uint8)
                img = np.transpose(img, (2, 0, 1))
                cam_imgs.append(img)
            cam_imgs = np.stack(cam_imgs, axis=0)

            # action chunk
            a_pos_full = np.asarray(h["/action/position"][()], dtype=np.float32)
            a_frc_full = np.asarray(h["/action/force"][()], dtype=np.float32)
            act_full = np.concatenate([a_pos_full, a_frc_full], axis=-1).astype(np.float32)

            end = min(valid_len, a_start + self.seq_len)
            seg = act_full[a_start:end]
            L = seg.shape[0]

            is_pad = np.zeros((self.seq_len,), dtype=np.bool_)
            if L < self.seq_len:
                is_pad[L:] = True

            # force history (same episode, same observation timeline)
            if self.return_force_history:
                force_full = np.asarray(h["/observations/force"][()], dtype=np.float32)
                if valid_len < force_full.shape[0]:
                    force_full = force_full[:valid_len]

                force_history = _build_force_history(
                    force_full=force_full,
                    current_ts=start_ts,
                    history_len=self.force_history_len,
                )

        # image -> [0,1]
        image_t = torch.from_numpy(cam_imgs).float() / 255.0

        # qpos normalize with qpos stats
        qpos_t = torch.from_numpy(qpos).float()
        qpos_t = (qpos_t - self.q_min) / self.q_den
        qpos_t = torch.clamp(qpos_t, 0.0, 1.0)

        # action normalize with action stats
        action_t = torch.zeros((self.seq_len, 9), dtype=torch.float32)
        if L > 0:
            seg_t = torch.from_numpy(seg).float()
            seg_t = (seg_t - self.a_min) / self.a_den
            seg_t = torch.clamp(seg_t, 0.0, 1.0)
            action_t[:L] = seg_t

        pad_t = torch.from_numpy(is_pad).bool()

        if not self.return_force_history:
            return image_t, qpos_t, action_t, pad_t

        # force history normalize with qpos force stats
        force_history_t = torch.from_numpy(force_history).float()   # (L,3)
        force_history_t = (force_history_t - self.f_min) / self.f_den
        force_history_t = torch.clamp(force_history_t, 0.0, 1.0)

        return image_t, qpos_t, action_t, pad_t, force_history_t