#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import re
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
from torch.utils.data import DataLoader

from .dataset import EpisodicStartDataset


_EPISODE_RE = re.compile(r"episode_(\d+)\.hdf5$")


def _episode_sort_key(path: str):
    name = os.path.basename(path)
    m = _EPISODE_RE.search(name)
    if m is not None:
        return int(m.group(1))
    return name


def _find_episode_files(dataset_dir: str) -> List[str]:
    pattern = os.path.join(dataset_dir, "episode_*.hdf5")
    files = glob.glob(pattern)
    files = sorted(files, key=_episode_sort_key)
    if len(files) == 0:
        raise FileNotFoundError(f"No episode_*.hdf5 found under: {dataset_dir}")
    return files


def _infer_is_sim_from_file(path: str) -> bool:
    """
    Backward-compatible helper.
    If the file has '/observations/is_sim', use it.
    Otherwise default to False.
    """
    try:
        with h5py.File(path, "r") as h:
            if "/observations/is_sim" in h:
                val = np.asarray(h["/observations/is_sim"][()])
                if np.ndim(val) == 0:
                    return bool(val.item())
                return bool(val[0])
    except Exception:
        pass
    return False


def _get_valid_len(is_pad: np.ndarray) -> int:
    idx = np.where(is_pad.astype(bool))[0]
    return int(idx[0]) if len(idx) > 0 else int(is_pad.shape[0])


def _compute_norm_stats_all(episode_files: List[str]) -> Dict[str, np.ndarray]:
    """
    Compute per-dimension min/max over all episodes.
    Current baseline behavior is kept:
    - qpos stats from observation position(6)+force(3)
    - action stats from action position(6)+force(3)
    """
    all_qpos = []
    all_action = []

    for path in episode_files:
        with h5py.File(path, "r") as h:
            is_pad_full = np.asarray(h["/observations/is_pad"][()], dtype=np.bool_)
            valid_len = _get_valid_len(is_pad_full)
            if valid_len <= 0:
                valid_len = int(is_pad_full.shape[0])

            obs_pos = np.asarray(h["/observations/position"][:valid_len], dtype=np.float32)
            obs_frc = np.asarray(h["/observations/force"][:valid_len], dtype=np.float32)
            qpos = np.concatenate([obs_pos, obs_frc], axis=-1).astype(np.float32)
            all_qpos.append(qpos)

            act_pos = np.asarray(h["/action/position"][:valid_len], dtype=np.float32)
            act_frc = np.asarray(h["/action/force"][:valid_len], dtype=np.float32)
            action = np.concatenate([act_pos, act_frc], axis=-1).astype(np.float32)
            all_action.append(action)

    qpos_cat = np.concatenate(all_qpos, axis=0)
    action_cat = np.concatenate(all_action, axis=0)

    stats = {
        "qpos_min": qpos_cat.min(axis=0).astype(np.float32),
        "qpos_max": qpos_cat.max(axis=0).astype(np.float32),
        "action_min": action_cat.min(axis=0).astype(np.float32),
        "action_max": action_cat.max(axis=0).astype(np.float32),
    }
    return stats


def _split_train_val(
    episode_files: List[str],
    seed: int = 0,
    val_ratio: float = 0.2,
) -> Tuple[List[str], List[str]]:
    n = len(episode_files)
    if n == 1:
        return episode_files, episode_files

    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(n)

    n_val = max(1, int(round(n * float(val_ratio))))
    n_val = min(n_val, n - 1)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_files = [episode_files[i] for i in train_idx]
    val_files = [episode_files[i] for i in val_idx]

    train_files = sorted(train_files, key=_episode_sort_key)
    val_files = sorted(val_files, key=_episode_sort_key)
    return train_files, val_files


def load_data(
    dataset_dir: str,
    camera_names: List[str],
    batch_size_train: Optional[int] = None,
    batch_size_val: Optional[int] = None,
    batch_size: Optional[int] = None,
    seq_len_train: Optional[int] = None,
    seq_len_val: Optional[int] = None,
    seq_len: Optional[int] = None,
    samples_per_episode: int = 50,
    seed: int = 0,
    val_ratio: float = 0.2,
    is_sim: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    drop_last_train: bool = True,
    drop_last_val: bool = False,
    force_history_len: int = 10,
    return_force_history: bool = False,
    use_force_history: Optional[bool] = None,
    **kwargs,
):
    """
    Backward-compatible loader.

    Force-history mode:
      - set return_force_history=True
      - or use_force_history=True

    Returns:
      train_loader, val_loader, norm_stats, data_meta
    """
    if batch_size_train is None:
        batch_size_train = batch_size if batch_size is not None else 8
    if batch_size_val is None:
        batch_size_val = batch_size if batch_size is not None else batch_size_train

    if seq_len_train is None:
        seq_len_train = seq_len if seq_len is not None else 100
    if seq_len_val is None:
        seq_len_val = seq_len if seq_len is not None else seq_len_train

    if use_force_history is not None:
        return_force_history = bool(use_force_history)

    episode_files = _find_episode_files(dataset_dir)

    if is_sim is None:
        is_sim = _infer_is_sim_from_file(episode_files[0])

    train_files, val_files = _split_train_val(
        episode_files=episode_files,
        seed=seed,
        val_ratio=val_ratio,
    )

    # current baseline behavior: stats over all episodes
    norm_stats = _compute_norm_stats_all(episode_files)

    train_dataset = EpisodicStartDataset(
        episode_files=train_files,
        camera_names=camera_names,
        norm_stats=norm_stats,
        seq_len=seq_len_train,
        samples_per_episode=samples_per_episode,
        seed=seed,
        is_sim=is_sim,
        force_history_len=force_history_len,
        return_force_history=return_force_history,
    )

    val_dataset = EpisodicStartDataset(
        episode_files=val_files,
        camera_names=camera_names,
        norm_stats=norm_stats,
        seq_len=seq_len_val,
        samples_per_episode=samples_per_episode,
        seed=seed + 999,
        is_sim=is_sim,
        force_history_len=force_history_len,
        return_force_history=return_force_history,
    )

    train_loader_kwargs = dict(
        batch_size=batch_size_train,
        shuffle=True,
        drop_last=drop_last_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader_kwargs = dict(
        batch_size=batch_size_val,
        shuffle=False,
        drop_last=drop_last_val,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = persistent_workers
        val_loader_kwargs["persistent_workers"] = persistent_workers

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    data_meta = {
        "N": len(episode_files),
        "is_sim": bool(is_sim),
        "train_files": len(train_files),
        "val_files": len(val_files),
        "camera_names": list(camera_names),
        "samples_per_episode": int(samples_per_episode),
        "seq_len_train": int(seq_len_train),
        "seq_len_val": int(seq_len_val),
        "return_force_history": bool(return_force_history),
        "force_history_len": int(force_history_len),
    }

    return train_loader, val_loader, norm_stats, data_meta