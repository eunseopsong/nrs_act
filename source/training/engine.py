#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from copy import deepcopy

import torch
from tqdm import tqdm

from common.utils import compute_dict_mean, set_seed
from models.policy import ACTPolicy, CNNMLPPolicy
from training.debug import debug_norm_once, make_grad_scaler, autocast_context
from training.plotting import plot_history


def make_policy(policy_class: str, policy_config: dict):
    if policy_class == "ACT":
        return ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        return CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError(policy_class)


def forward_pass(batch, policy, device):
    image, qpos, action, is_pad = batch
    image = image.to(device, non_blocking=True)
    qpos = qpos.to(device, non_blocking=True)
    action = action.to(device, non_blocking=True)
    is_pad = is_pad.to(device, non_blocking=True)
    return policy(qpos, image, action, is_pad)


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
    scaler = make_grad_scaler(use_amp, device)

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
                with autocast_context(use_amp, device):
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
            with autocast_context(use_amp, device):
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