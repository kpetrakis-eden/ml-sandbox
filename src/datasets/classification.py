from pathlib import Path
import math
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler,Subset, BatchSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as v2

from src.utils.config import DataConfig
from hydra.utils import instantiate

class DataFactory:
  def __init__(self, cfg: DataConfig, generator: torch.Generator):
    self.cfg = cfg
    self.g = generator
    # Construct a visualization dataloader with that many samples from each class, to draw prediction dynamics on.
    self.viz_samples_per_class = {
      0: 10,
      1: 10,
      2: 10,
      3: 60,
      4: 50,
      5: 60,
    }
    train_transforms = [v2.ToImage(), v2.Resize((64,64))]
    dev_transforms = [v2.ToImage(), v2.Resize((64,64))]
    if cfg.augmentation is not None:
      train_transforms.extend([instantiate(t) for t in cfg.augmentation])

    train_transforms.append(v2.ToDtype(torch.float32, scale=True))
    dev_transforms.append(v2.ToDtype(torch.float32, scale=True))
    if cfg.normalization is not None:
      train_transforms.append(
        v2.Normalize(
          mean = self.cfg.normalization.mean,
          std = self.cfg.normalization.std
        )
      )
      dev_transforms.append(
        v2.Normalize(
          mean = self.cfg.normalization.mean,
          std = self.cfg.normalization.std
        )
      )
    self.train_transforms = v2.Compose(train_transforms)
    self.dev_transforms = v2.Compose(dev_transforms)
    print(f"train transforms: {self.train_transforms}")
    print(f"dev transforms: {self.dev_transforms}")

    self.train_ds = None
    self.dev_ds = None
    self.sampler = None
    self.viz_subset:Subset = None

  def build_datasets(self):
    self.train_ds = ImageFolder(self.cfg.root / "train", transform=self.train_transforms)
    self.dev_ds = ImageFolder(self.cfg.root / "dev", transform=self.dev_transforms)
    return self

  def build_sampler(self):
    if self.cfg.sampling not in ["default", "weighted", "balanced"]:
      raise ValueError(f"Sampling method {self.cfg.sampling} not supported.")
    if self.cfg.sampling == "weighted":
      class_counts = np.bincount(self.train_ds.targets)
      class_weights = 1.0 / class_counts
      sample_weights = class_weights[self.train_ds.targets]

      self.sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    elif self.cfg.sampling == "balanced":
      self.sampler = BalancedBatchSampler(self.train_ds.targets, self.cfg.batch_size, self.g)
    else:
      self.sampler = None

    return self
    '''
    if self.cfg.sampling != "weighted":
      self.sampler = None
      return self

    class_counts = np.bincount(self.train_ds.targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[self.train_ds.targets]

    self.sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return self
    '''

  def build_loaders(self):
    if self.train_ds is None or self.dev_ds is None:
      raise RuntimeError("Call build_datasets() first")

    if self.sampler is None:
      train_loader = DataLoader(self.train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, generator=self.g)
    else:
      if isinstance(self.sampler, WeightedRandomSampler):
        train_loader = DataLoader(self.train_ds, batch_size=self.cfg.batch_size, sampler=self.sampler, num_workers=self.cfg.num_workers, generator=self.g)
      elif isinstance(self.sampler, BalancedBatchSampler):
        train_loader = DataLoader(self.train_ds, batch_sampler=self.sampler, num_workers=self.cfg.num_workers, generator=self.g)

    dev_loader = DataLoader(self.dev_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, generator=self.g)

    return train_loader, dev_loader


  def build_viz_loader(self):
    '''
    return Dataloader, which is a subset of the dev_dataset to log predictions dynamics on
    '''
    viz_loader = DataLoader(self.viz_subset, batch_size=self.cfg.batch_size, shuffle=False)
    return viz_loader

  def build_viz_subset(self, seed:int):
    if self.dev_ds is None:
      raise RuntimeError("Dev dataset is None, build_datasets() first")

    rng = np.random.default_rng(seed)
    class_to_indices = defaultdict(list)
    # ImageFolder exposes labels in dataset.targets
    for idx, label in enumerate(self.dev_ds.targets):
      class_to_indices[label].append(idx)

    subset_indices = []
    for label, indices in class_to_indices.items():
      n_select = min(self.viz_samples_per_class.get(label, 5), len(indices))
      selected = rng.choice(indices, size=n_select, replace=False)

      # print(f"{label}: {selected}")
      subset_indices.extend(selected)

    rng.shuffle(subset_indices)
    self.viz_subset = Subset(self.dev_ds, subset_indices)
    return self

class BalancedBatchSampler(BatchSampler):
  def __init__(self, targets, batch_size, g:torch.Generator):

    self.targets = np.array(targets, dtype=np.int64)
    self.batch_size = batch_size
    self.classes = np.unique(self.targets).tolist()
    self.n_classes = len(self.classes)
    self.g = g

    self.samples_per_class = batch_size // self.n_classes

    assert self.targets.ndim == 1, "targets should be 1d"
    assert self.batch_size % self.n_classes == 0, "batch size should be a multiple of num_classes"

    # targets has to be a np.array for this to be safe, otherwise is wrong
    self.class_indices = { c: np.nonzero(self.targets == c)[0] for c in self.classes }

     # Minority class size
    self.minority_size = min( len(v) for v in self.class_indices.values())
    self.n_batches = math.ceil(self.minority_size / self.samples_per_class)

  def __len__(self):
    return self.n_batches

  def __iter__(self):
    shuffled = {} # same as class_indices , but shuffled
    for c, idxs in self.class_indices.items():
      perm = torch.randperm(len(idxs), generator=self.g)
      shuffled[c] = idxs.copy()[perm]

    for batch_idx in range(self.n_batches):
      batch = []

      start = batch_idx*self.samples_per_class
      end = start + self.samples_per_class # for the last batch and minority class this is > len(idxes) but is ok, i just keep the remaining

      for c in self.classes:
        batch.extend(shuffled[c][start:end].tolist())

      bperm = torch.randperm(len(batch), generator=self.g)
      batch = [batch[i] for i in bperm]
      yield batch

# v2.Normalize(mean=[0.4292, 0.5389, 0.3654], std=[0.1860, 0.2121, 0.1966])
# [0.4291510581970215, 0.5389042496681213, 0.3654465973377228], [0.18595948815345764, 0.21214619278907776, 0.19659456610679626]
# 
# Those weights were computed on all the data,before the split..
# ** THEY MUST NOT BE USED FOR training/dev after splitting.. I have to calculate only the training ones!!
# 
# the new values after the split, are not so different!!