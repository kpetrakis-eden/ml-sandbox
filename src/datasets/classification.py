from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as v2

from src.utils.config import DataConfig
from hydra.utils import instantiate

class DataFactory:
  def __init__(self, cfg: DataConfig, generator: torch.Generator):
    self.cfg = cfg
    self.g = generator
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

  def build_datasets(self):
    self.train_ds = ImageFolder(self.cfg.root / "train", transform=self.train_transforms)
    self.dev_ds = ImageFolder(self.cfg.root / "dev", transform=self.dev_transforms)
    return self

  def build_sampler(self):
    if self.cfg.sampling != "weighted":
      self.sampler = None
      return self

    class_counts = np.bincount(self.train_ds.targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[self.train_ds.targets]

    self.sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return self

  def build_loaders(self):
    if self.train_ds is None or self.dev_ds is None:
      raise RuntimeError("Call build_datasets() first")

    if self.sampler is None:
      train_loader = DataLoader(self.train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, generator=self.g)
    else:
      train_loader = DataLoader(self.train_ds, batch_size=self.cfg.batch_size, sampler=self.sampler, num_workers=self.cfg.num_workers, generator=self.g)

    dev_loader = DataLoader(self.dev_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, generator=self.g)

    return train_loader, dev_loader

'''
@dataclass
class NormalizationConfig:
  enabled: bool
  mean: Optional[list] = None
  std: Optional[list] = None

@dataclass
class DataConfig:
  root: Path
  num_workers: int
  batch_size: int
  sampling: str = "default" # "default" | "weighted"
  normalization: NormalizationConfig = None

  def __post_init__(self):
    self.root = Path(self.root)
    self.normalization = NormalizationConfig(**self.normalization)

def get_dataloaders(data_dir: Path, batch_size: int, g: torch.Generator, sampler:Sampler = None):
  # NOTE: if I add random transforms then I need seed_worker for determinism in dataloading
  transforms = v2.Compose([
    v2.Resize((64, 64)),
    v2.ToImage(),
    # v2.CenterCrop(),

    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.4291, 0.5388, 0.3654], std=[0.1859, 0.2121, 0.1966])
    # [0.42910709977149963, 0.5388360023498535, 0.36541175842285156] , std: [0.1859438121318817, 0.21211254596710205, 0.19656462967395782]
  ])

  data_dir = Path(data_dir)
  train_dataset = ImageFolder(data_dir / "train", transform=transforms)
  dev_dataset = ImageFolder(data_dir / "dev", transform=transforms)

  if sampler is None:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, generator=g)
  else:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=12, generator=g)

  dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=12, generator=g)

  return train_dataloader, dev_dataloader

# v2.Normalize(mean=[0.4292, 0.5389, 0.3654], std=[0.1860, 0.2121, 0.1966])
# [0.4291510581970215, 0.5389042496681213, 0.3654465973377228], [0.18595948815345764, 0.21214619278907776, 0.19659456610679626]
# 
# Those weights were computed on all the data,before the split..
# ** THEY MUST NOT BE USED FOR training/dev after splitting.. I have to calculate only the training ones!!
# 
# the new values after the split, are not so different!!
'''