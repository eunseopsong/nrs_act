#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from typing import Any, Dict, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detach_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v.detach()
        else:
            out[k] = v
    return out


def compute_dict_mean(dict_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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