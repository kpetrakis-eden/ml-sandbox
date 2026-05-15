import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.extra import compute_class_weights
from src.utils.config import LossConfig
from .focal import FocalLoss

def get_loss_fn(loss_cfg: LossConfig, dataloader:DataLoader, device:torch.device):
  loss_name = loss_cfg.name
  if loss_name == "cross-entropy":
    return nn.CrossEntropyLoss()
  elif loss_name == "cross-entropy-weighted":
    class_weights = compute_class_weights(dataloader, device)
    return nn.CrossEntropyLoss(weight=class_weights)
  elif loss_name == "focal":
    # gamma = loss_config['loss']['focal_gamma']
    # reduction = config['loss']['focal_reduction']
    gamma = loss_cfg.focal_gamma
    reduction = loss_cfg.focal_reduction
    return FocalLoss(alpha=class_weights, gamma=gamma, reduction=reduction)
  else:
    raise RuntimeError(f"{loss_name} not supported.")