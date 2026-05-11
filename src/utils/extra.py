# TODO: name this file appropriately
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import Counter

def compute_class_weights(dataloader:DataLoader, device:torch.device):
  '''
  there are different ways to compute class weights.. Check notebooks/Classification
  '''
  class_counts = np.bincount(dataloader.dataset.targets)
  class_weights = 1.0 / class_counts
  class_weights = class_weights / class_weights.sum() * len(class_weights)
  class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

  return class_weights


def build_weighted_sampler():
  raise NotImplementedError