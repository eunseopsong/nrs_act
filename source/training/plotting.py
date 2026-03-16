#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np


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