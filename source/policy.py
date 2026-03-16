import os
import sys
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms


# -----------------------------
# source/act/detr 경로 보정
# policy.py 하나만 사용하도록 유지
# -----------------------------
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ACT_DIR = os.path.join(_CURRENT_DIR, "act")
_DETR_PARENT = _ACT_DIR  # contains package folder: detr/

if _DETR_PARENT not in sys.path:
    sys.path.insert(0, _DETR_PARENT)

try:
    from detr.main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
except ImportError as e:
    raise ImportError(
        f"[source/policy.py] 'detr' 패키지를 찾을 수 없습니다.\n"
        f"현재 작업 디렉터리: {os.getcwd()}\n"
        f"기대한 경로: {_DETR_PARENT}\n"
        f"원래 에러: {e}"
    )


class ACTPolicy(nn.Module):
    """ACT 기반 정책 네트워크"""

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]

        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        print(f"[ACTPolicy] KL Weight = {self.kl_weight}")

    def forward(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        image = self._normalize(image)

        # training
        if actions is not None:
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, env_state, actions, is_pad
            )

            total_kld, _, _ = kl_divergence(mu, logvar)

            all_l1 = F.l1_loss(actions, a_hat, reduction="none")   # (B,T,D)
            valid_mask = (~is_pad).unsqueeze(-1).float()           # (B,T,1)

            # padded timesteps 제외한 평균
            valid_count = valid_mask.sum().clamp_min(1.0) * actions.shape[-1]
            l1 = (all_l1 * valid_mask).sum() / valid_count

            loss_dict = {
                "l1": l1,
                "kl": total_kld[0],
            }
            loss_dict["loss"] = loss_dict["l1"] + self.kl_weight * loss_dict["kl"]
            return loss_dict

        # inference
        a_hat, _, _ = self.model(qpos, image, env_state)
        return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    """단순 CNN+MLP 정책"""

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer

        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        image = self._normalize(image)

        if actions is not None:
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            return {"mse": mse, "loss": mse}

        a_hat = self.model(qpos, image, env_state)
        return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    """ACT에서 사용하는 KL 계산"""
    batch_size = mu.size(0)
    assert batch_size != 0

    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
