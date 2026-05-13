import torch.nn as nn
from .focal import FocalLoss

def get_loss_fn(config, class_weights=None):
  loss_name = config["loss_name"]
  if loss_name == "cross-entropy":
    return nn.CrossEntropyLoss(weight=class_weights)

  elif loss_name == "focal":
    gamma = config['loss']['focal_gamma']
    return FocalLoss(alpha=class_weights, gamma=gamma, reduction='mean')
