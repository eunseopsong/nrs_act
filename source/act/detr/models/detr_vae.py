# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR-VAE model (UR10e: position+force + multi-camera image -> position+force)

IMPORTANT:
- qpos/action normalization is done in Dataset (mean/std).
- image normalization (ImageNet mean/std) is done in policy.py.
- This model does NOT do extra normalization.
"""

import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
import numpy as np


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """
    Default:
      obs dim    = 9  (pos6 + force3)
      action dim = 9  (pos6 + force3)
    """
    def __init__(self, backbones, transformer, encoder,
                 obs_dim, action_dim, num_queries, camera_names):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model

        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(obs_dim, hidden_dim)
        else:
            self.backbones = None
            self.input_proj_robot_state = nn.Linear(obs_dim, hidden_dim)

        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)

        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_joint_proj  = nn.Linear(obs_dim, hidden_dim)

        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer("pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))

        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def forward(self, qpos, image, env_state=None, actions=None, is_pad=None):
        # accept both (B,9) or (B,1,9) etc.
        if qpos.dim() == 3:
            qpos = qpos[:, 0, :]
        # accept both (B,K,3,H,W) or (B,1,K,3,H,W)
        if image.dim() == 6:
            image = image[:, 0, ...]

        is_training = actions is not None
        bs, _ = qpos.shape

        if is_training:
            action_embed = self.encoder_action_proj(actions)         # (B,T,H)
            qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)  # (B,1,H)
            cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)  # (B,1,H)

            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)  # (B, 2+T, H)
            encoder_input = encoder_input.permute(1, 0, 2)  # (2+T, B, H)

            cls_joint_is_pad = torch.full((bs, 2), False, device=qpos.device)
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (B,2+T)

            pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)  # (2+T,1,H)

            encoder_output = self.encoder(
                encoder_input,
                pos=pos_embed,
                src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0, :, :]  # CLS token output (B,H)

            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32, device=qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            all_cam_features, all_cam_pos = [], []

            for cam_id, cam_name in enumerate(self.camera_names):
                cam_img = image[:, cam_id]                 # (B,3,H,W)
                features, pos = self.backbones[0](cam_img) # NOTE: your code uses backbones[0] shared

                features = features[0]
                pos = pos[0]

                proj_feat = self.input_proj(features)
                all_cam_features.append(proj_feat)
                all_cam_pos.append(pos)

            # concat along width
            src = torch.cat(all_cam_features, dim=3)
            pos = torch.cat(all_cam_pos, dim=3)

            proprio_input = self.input_proj_robot_state(qpos)

            hs = self.transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight
            )[0]
        else:
            qpos_emb = self.input_proj_robot_state(qpos)
            hs = self.transformer(
                qpos_emb, None,
                self.query_embed.weight,
                self.additional_pos_embed.weight
            )[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


class CNNMLP(nn.Module):
    def __init__(self, backbones, obs_dim, action_dim, camera_names):
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, action_dim)
        self.backbones = nn.ModuleList(backbones)

        backbone_down_projs = []
        for backbone in backbones:
            down_proj = nn.Sequential(
                nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                nn.Conv2d(128, 64, kernel_size=5),
                nn.Conv2d(64, 32, kernel_size=5)
            )
            backbone_down_projs.append(down_proj)
        self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

        mlp_in_dim = 768 * len(backbones) + obs_dim
        self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=512, output_dim=action_dim, hidden_depth=2)

    def forward(self, qpos, image, env_state=None, actions=None):
        if qpos.dim() == 3:
            qpos = qpos[:, 0, :]
        if image.dim() == 6:
            image = image[:, 0, ...]

        bs, _ = qpos.shape
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, _ = self.backbones[cam_id](image[:, cam_id])
            features = features[0]
            cam_feat = self.backbone_down_projs[cam_id](features)
            all_cam_features.append(cam_feat.reshape([bs, -1]))

        flattened = torch.cat(all_cam_features, axis=1)
        features = torch.cat([flattened, qpos], axis=1)
        return self.mlp(features)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
    for _ in range(hidden_depth - 1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
    mods.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*mods)


def build_encoder(args):
    d_model = args.hidden_dim
    dropout = getattr(args, "dropout", 0.1)
    nhead = args.nheads
    dim_feedforward = args.dim_feedforward
    num_encoder_layers = args.enc_layers
    normalize_before = getattr(args, "pre_norm", False)
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(
        d_model, nhead, dim_feedforward, dropout, activation, normalize_before
    )
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    return TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)


def build(args):
    # allow override from args
    obs_dim = getattr(args, "state_dim", 9)
    action_dim = getattr(args, "action_dim", 9)

    backbones = [build_backbone(args)]
    transformer = build_transformer(args)
    encoder = build_encoder(args)

    model = DETRVAE(
        backbones, transformer, encoder,
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_params / 1e6,))
    return model


def build_cnnmlp(args):
    obs_dim = getattr(args, "state_dim", 9)
    action_dim = getattr(args, "action_dim", 9)

    backbones = [build_backbone(args) for _ in args.camera_names]
    model = CNNMLP(backbones, obs_dim=obs_dim, action_dim=action_dim, camera_names=args.camera_names)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_params / 1e6,))
    return model
