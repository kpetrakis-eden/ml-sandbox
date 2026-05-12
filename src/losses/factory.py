import torch.nn as nn
from .focal import FocalLoss

def get_loss_fn(config, class_weights=None):
  loss_name = config["loss_name"]
  if loss_name == "cross-entropy":
    return nn.CrossEntropyLoss(weight=class_weights)

  elif loss_name == "focal":
    # raise NotImplementedError
    return FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')
