import torch.nn as nn

def get_loss_fn(config, weights=None):
  loss_name = config["loss_name"]
  if loss_name == "cross-entropy":
    return nn.CrossEntropyLoss(weight=weights)

  elif loss_name == "focal":
    raise NotImplementedError
