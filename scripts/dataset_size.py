import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as v2
import torch.nn as nn
from pathlib import Path
import numpy as np

DATASET = "classification-merged-pink-purple-v2"
CLASSIFICATION_ROOT =  Path.cwd() / f"data/processed/{DATASET}"

TRAIN_DIR = CLASSIFICATION_ROOT / "train"
DEV_DIR = CLASSIFICATION_ROOT / "dev"

print(TRAIN_DIR.exists())

if __name__ == "__main__":
  transforms = v2.Identity()

  g = torch.Generator().manual_seed(0)
  train_dataset = ImageFolder(TRAIN_DIR, transform=transforms)
  dev_dataset = ImageFolder(DEV_DIR, transform=transforms)
  # zero_dataset = ZeroDataset(dataset) # to verify it doesn't learn on zero input

  # ================================================================
  class_counts_train = np.bincount(train_dataset.targets)
  class_counts_dev = np.bincount(dev_dataset.targets)
  print(f"Train instances per class: {class_counts_train}, dev instances per class: {class_counts_dev}")