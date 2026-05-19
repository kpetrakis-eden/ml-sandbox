from src.utils.config import OptimizerConfig, SchedulerConfig
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, StepLR


def get_optimizer(cfg:OptimizerConfig, model:nn.Module):
  if cfg.name == "adamw":
    return AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

  elif cfg.name == "sgd":
    if cfg.momentum and cfg.nesterov:
      return SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov)
    else:
      return SGD(model.parameters(), lr=cfg.lr)


def get_scheduler(cfg:SchedulerConfig, optimizer:torch.optim):
  # no scheduler specified in config
  if cfg is None:
    return None
  else:
    match cfg.name:
      case "cosine":
        return CosineAnnealingLR(optimizer=optimizer, T_max=cfg.T_max) 
      case "linear":
        return LinearLR(opitmizer=optimizer, start_factor=cfg.start_factor, end_factor=cfg.end_factor, total_iterls=cfg.total_iters)
      case "step":
        return StepLR(optimizer=optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
      case _:
        return None