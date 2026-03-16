#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import List

import torch
import torch.distributed as dist
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter


def is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _build_resnet_backbone(name: str, dilation: bool, pretrained_backbone: bool, norm_layer):
    name = str(name)

    weights = None
    if pretrained_backbone and is_main_process():
        if name == "resnet18":
            weights = torchvision.models.ResNet18_Weights.DEFAULT
        elif name == "resnet34":
            weights = torchvision.models.ResNet34_Weights.DEFAULT
        elif name == "resnet50":
            weights = torchvision.models.ResNet50_Weights.DEFAULT
        elif name == "resnet101":
            weights = torchvision.models.ResNet101_Weights.DEFAULT
        else:
            weights = None

    ctor = getattr(torchvision.models, name)
    return ctor(
        weights=weights,
        replace_stride_with_dilation=[False, False, dilation],
        norm_layer=norm_layer,
    )


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor: torch.Tensor):
        x = tensor
        not_mask = torch.ones_like(x[:, 0:1, :, :], dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, 0, :, :, None] / dim_t
        pos_y = y_embed[:, 0, :, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor: torch.Tensor):
        x = tensor
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)

        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pos = torch.cat(
            [
                x_emb.unsqueeze(0).repeat(h, 1, 1),
                y_emb.unsqueeze(1).repeat(1, w, 1),
            ],
            dim=-1,
        ).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    n_steps = args.hidden_dim // 2
    if args.position_embedding in ("v2", "sine"):
        return PositionEmbeddingSine(n_steps, normalize=True)
    if args.position_embedding in ("v3", "learned"):
        return PositionEmbeddingLearned(n_steps)
    raise ValueError(f"not supported {args.position_embedding}")


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()

        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

        _ = train_backbone

    def forward(self, tensor: torch.Tensor):
        xs = self.body(tensor)
        return xs


class Backbone(BackboneBase):
    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool, pretrained_backbone: bool = True):
        backbone = _build_resnet_backbone(
            name=name,
            dilation=dilation,
            pretrained_backbone=pretrained_backbone,
            norm_layer=FrozenBatchNorm2d,
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor: torch.Tensor):
        xs = self[0](tensor)
        out: List[torch.Tensor] = []
        pos = []

        for _, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = getattr(args, "masks", False)
    pretrained_backbone = getattr(args, "pretrained_backbone", True)

    backbone = Backbone(
        args.backbone,
        train_backbone=train_backbone,
        return_interm_layers=return_interm_layers,
        dilation=args.dilation,
        pretrained_backbone=pretrained_backbone,
    )
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model