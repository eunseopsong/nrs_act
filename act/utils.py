# utils.py
# Minimal utilities used by training/eval scripts (compatible with your new pipeline)

import os
import random
from typing import Any, Dict, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism (may reduce speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detach_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Detach all tensor values (shallow)."""
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
    Returns torch tensors on CPU (so float() works).
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
                # ensure scalar
                if v.numel() != 1:
                    v = v.mean()
            else:
                v = torch.tensor(float(v), dtype=torch.float32)

            vals.append(v)

        if len(vals) == 0:
            continue

        out[k] = torch.stack(vals).mean()

    return out
