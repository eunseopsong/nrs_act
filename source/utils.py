#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py


# -----------------------------
# Basic utils
# -----------------------------
def set_seed(seed: int) -> None:
    """
    Set seeds for reproducibility.
    Keep behavior lightweight (do not force cudnn deterministic here).
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detach_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detach all tensor values (shallow).
    """
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v.detach()
        else:
            out[k] = v
    return out


def compute_dict_mean(dict_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Compute mean over a list of metric dicts.
    Values can be torch tensors or floats/ints.
    Returns torch tensors on CPU.
    """
    if len(dict_list) == 0:
        return {}

    keys = set()
    for d in dict_list:
        keys.update(d.keys())

    out: Dict[str, torch.Tensor] = {}
    for k in sorted(keys):
        vals = []
        for d in dict_list:
            if k not in d:
                continue

            v = d[k]
            if torch.is_tensor(v):
                v = v.detach().float().cpu()
                if v.numel() != 1:
                    v = v.mean()
            else:
                v = torch.tensor(float(v), dtype=torch.float32)

            vals.append(v)

        if len(vals) == 0:
            continue

        out[k] = torch.stack(vals).mean()

    return out


# -----------------------------
# HDF5 helpers
# -----------------------------
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


# -----------------------------
# Normalization stats (MIN-MAX per-dim -> [0,1])
# -----------------------------
def compute_norm_stats_all(episode_files: List[str]) -> Dict[str, np.ndarray]:
    """
    Compute min/max for:
      qpos   = [position(6), force(3)] => (9,)
      action = [position(6), force(3)] => (9,)
    over all valid (non-pad) timesteps of all given episodes.

    Returned stats:
      qpos_min, qpos_max, action_min, action_max (float32, shape (9,))
    """
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

            qv = q[valid]  # (N,9)
            av = a[valid]  # (N,9)

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
    """
    action_norm: (...,9) normalized in [0,1]
    return:      (...,9) real scale
    """
    if "action_min" in stats and "action_max" in stats:
        mn = stats["action_min"].reshape((1,) * (action_norm.ndim - 1) + (9,))
        mx = stats["action_max"].reshape((1,) * (action_norm.ndim - 1) + (9,))
        return action_norm * (mx - mn) + mn

    # backward compatibility (old mean/std)
    mu = stats["action_mean"].reshape((1,) * (action_norm.ndim - 1) + (9,))
    sd = stats["action_std"].reshape((1,) * (action_norm.ndim - 1) + (9,))
    return action_norm * sd + mu


# -----------------------------
# Dataset (EpisodicStartDataset)
# -----------------------------
class EpisodicStartDataset(Dataset):
    """
    FAST start-ts sampling dataset.

    Returns:
      image : (K,3,H,W) float [0,1]
      qpos  : (9,)      min-max normalized to [0,1] per-dim
      action: (T,9)     min-max normalized to [0,1] per-dim
      is_pad: (T,)      bool

    IMPORTANT:
      __len__ = num_episodes * samples_per_episode
      -> restores "many iterations per epoch"
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

            # qpos(9) = pos6 + force3
            pos = np.asarray(h["/observations/position"][start_ts], dtype=np.float32)
            frc = np.asarray(h["/observations/force"][start_ts], dtype=np.float32)
            qpos = np.concatenate([pos, frc], axis=-1).astype(np.float32)

            # images at start_ts only (KEEP multi-cam: (K,3,H,W))
            img_grp = h["/observations/images"]
            cam_imgs = []
            for cam in self.camera_names:
                cam_key = _resolve_cam_key(img_grp, cam)
                img = np.asarray(img_grp[cam_key][start_ts], dtype=np.uint8)  # (H,W,3)
                img = np.transpose(img, (2, 0, 1))  # (3,H,W)
                cam_imgs.append(img)
            cam_imgs = np.stack(cam_imgs, axis=0)  # (K,3,H,W)

            # action full (T,9)
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


# -----------------------------
# load_data() (KEEP existing episode_*.hdf5 search)
# -----------------------------
def load_data(
    dataset_dir: str,
    num_episodes: int,
    camera_names: List[str],
    batch_size_train: int,
    batch_size_val: int,
    seq_len_train: int,
    seq_len_val: int,
    seed: int = 0,
    samples_per_episode: int = 50,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    drop_last_train: bool = True,
) -> Tuple[DataLoader, DataLoader, Dict[str, np.ndarray], Dict]:
    """
    - list episode_*.hdf5 (DIRECT under dataset_dir)
    - 80/20 split
    - compute min/max stats
    - create datasets/loaders
    - return (train_loader, val_loader, stats, meta)
    """
    ep_files = sorted(glob.glob(os.path.join(dataset_dir, "episode_*.hdf5")))
    if len(ep_files) == 0:
        raise FileNotFoundError(f"No episode_*.hdf5 found in {dataset_dir}")

    if int(num_episodes) > 0:
        N = min(int(num_episodes), len(ep_files))
        ep_files = ep_files[:N]
    else:
        N = len(ep_files)

    # sim attr (optional)
    is_sim = False
    try:
        with h5py.File(ep_files[0], "r") as h0:
            if "sim" in h0.attrs:
                is_sim = bool(h0.attrs["sim"])
    except Exception:
        is_sim = False

    # split 80/20
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_train = max(1, int(0.8 * N))
    train_files = [ep_files[i] for i in perm[:n_train]]
    val_files = [ep_files[i] for i in perm[n_train:]] or train_files[:max(1, len(train_files) // 5)]

    # NOTE:
    # Keep current behavior for compatibility.
    # If you want stricter evaluation later, change this to train_files only.
    stats = compute_norm_stats_all(train_files + val_files)

    train_ds = EpisodicStartDataset(
        train_files,
        camera_names,
        stats,
        seq_len=seq_len_train,
        samples_per_episode=samples_per_episode,
        seed=seed,
        is_sim=is_sim,
    )
    val_ds = EpisodicStartDataset(
        val_files,
        camera_names,
        stats,
        seq_len=seq_len_val,
        samples_per_episode=max(1, samples_per_episode // 5),
        seed=seed + 123,
        is_sim=is_sim,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        drop_last=drop_last_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        drop_last=False,
    )

    meta = dict(
        N=N,
        is_sim=is_sim,
        train_files=len(train_files),
        val_files=len(val_files),
        camera_names=list(camera_names),
        samples_per_episode=samples_per_episode,
        seq_len_train=seq_len_train,
        seq_len_val=seq_len_val,
    )
    return train_loader, val_loader, stats, meta
