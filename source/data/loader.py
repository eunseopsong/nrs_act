#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
from typing import Dict, List, Tuple

import h5py
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import EpisodicStartDataset
from data.normalization import compute_norm_stats_all


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
    ep_files = sorted(glob.glob(os.path.join(dataset_dir, "episode_*.hdf5")))
    if len(ep_files) == 0:
        raise FileNotFoundError(f"No episode_*.hdf5 found in {dataset_dir}")

    if int(num_episodes) > 0:
        N = min(int(num_episodes), len(ep_files))
        ep_files = ep_files[:N]
    else:
        N = len(ep_files)

    is_sim = False
    try:
        with h5py.File(ep_files[0], "r") as h0:
            if "sim" in h0.attrs:
                is_sim = bool(h0.attrs["sim"])
    except Exception:
        is_sim = False

    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_train = max(1, int(0.8 * N))
    train_files = [ep_files[i] for i in perm[:n_train]]
    val_files = [ep_files[i] for i in perm[n_train:]] or train_files[: max(1, len(train_files) // 5)]

    # 기존 동작 유지
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