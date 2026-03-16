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


class EpisodicStartDataset(Dataset):
    """
    Returns:
      image : (K,3,H,W) float [0,1]
      qpos  : (9,)
      action: (T,9)
      is_pad: (T,)
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
    ):
        self.files = list(episode_files)
        self.camera_names = list(camera_names)
        self.seq_len = int(seq_len)
        self.samples_per_episode = int(max(1, samples_per_episode))
        self.seed = int(seed)
        self.is_sim = bool(is_sim)

        self.q_min = torch.tensor(norm_stats["qpos_min"], dtype=torch.float32)
        self.q_max = torch.tensor(norm_stats["qpos_max"], dtype=torch.float32)
        self.a_min = torch.tensor(norm_stats["action_min"], dtype=torch.float32)
        self.a_max = torch.tensor(norm_stats["action_max"], dtype=torch.float32)

        eps = 1e-6
        self.q_den = torch.clamp(self.q_max - self.q_min, min=eps)
        self.a_den = torch.clamp(self.a_max - self.a_min, min=eps)

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

            pos = np.asarray(h["/observations/position"][start_ts], dtype=np.float32)
            frc = np.asarray(h["/observations/force"][start_ts], dtype=np.float32)
            qpos = np.concatenate([pos, frc], axis=-1).astype(np.float32)

            img_grp = h["/observations/images"]
            cam_imgs = []
            for cam in self.camera_names:
                cam_key = _resolve_cam_key(img_grp, cam)
                img = np.asarray(img_grp[cam_key][start_ts], dtype=np.uint8)
                img = np.transpose(img, (2, 0, 1))
                cam_imgs.append(img)
            cam_imgs = np.stack(cam_imgs, axis=0)

            a_pos_full = np.asarray(h["/action/position"][()], dtype=np.float32)
            a_frc_full = np.asarray(h["/action/force"][()], dtype=np.float32)
            act_full = np.concatenate([a_pos_full, a_frc_full], axis=-1).astype(np.float32)

            end = min(valid_len, a_start + self.seq_len)
            seg = act_full[a_start:end]
            L = seg.shape[0]

            is_pad = np.zeros((self.seq_len,), dtype=np.bool_)
            if L < self.seq_len:
                is_pad[L:] = True

        image_t = torch.from_numpy(cam_imgs).float() / 255.0
        qpos_t = torch.from_numpy(qpos).float()

        qpos_t = (qpos_t - self.q_min) / self.q_den
        qpos_t = torch.clamp(qpos_t, 0.0, 1.0)

        action_t = torch.zeros((self.seq_len, 9), dtype=torch.float32)
        if L > 0:
            seg_t = torch.from_numpy(seg).float()
            seg_t = (seg_t - self.a_min) / self.a_den
            seg_t = torch.clamp(seg_t, 0.0, 1.0)
            action_t[:L] = seg_t

        pad_t = torch.from_numpy(is_pad).bool()

        return image_t, qpos_t, action_t, pad_t