from src.utils.config import OptimizerConfig, SchedulerConfig
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD


def get_optimizer(cfg:OptimizerConfig, model:nn.Module):
  if cfg.name == "adamw":
    return AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

  elif cfg.name == "sgd":
    if cfg.momentum and cfg.nesterov:
      return SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov)
    else:
      return SGD(model.parameters(), lr=cfg.lr)


def get_scheduler(cfg:SchedulerConfig, optimizer:torch.optim):
  raise NotImplementedError