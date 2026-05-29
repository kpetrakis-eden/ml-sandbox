# TODO: name this file appropriately
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.utils.config import MLFlowConfig
import torchvision.transforms.v2.functional as V2F
import mlflow

class PadToSquare:
  def __init__(self, padding_mode:str='edge', fill:Optional[int]=0):
    self.fill = fill
    self.padding_mode = padding_mode

  def __call__(self, img):
    h, w = img.shape[-2:]
    max_wh = max(w, h)

    pad_left = (max_wh - w) // 2
    pad_right = max_wh - w - pad_left
    pad_top = (max_wh - h) // 2
    pad_bottom = max_wh - h - pad_top

    padding = [pad_left, pad_top, pad_right, pad_bottom]
    return V2F.pad(img, padding, padding_mode=self.padding_mode) # faster?
    # return v2.Pad(padding, padding_mode=self.padding_mode)(img)

  def __repr__(self):
    return (f"{self.__class__.__name__}(" f"padding_mode={self.padding_mode!r}, " f"fill={self.fill})")


def compute_class_weights(dataloader:DataLoader, device:torch.device):
  '''
  there are different ways to compute class weights.. Check notebooks/Classification
  '''
  class_counts = np.bincount(dataloader.dataset.targets)
  class_weights = 1.0 / class_counts
  class_weights = class_weights / class_weights.sum() * len(class_weights)
  class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

  return class_weights


def set_or_create_experiment(cfg: MLFlowConfig):
  mlflow.set_tracking_uri(f"sqlite:///{cfg.tracking_uri.resolve()}")
  experiment = mlflow.get_experiment_by_name(cfg.name)
  if experiment is None:
    experiment_id = mlflow.create_experiment(name=cfg.name, artifact_location=cfg.artifact_location.resolve().as_uri())
  else:
    experiment_id = experiment.experiment_id

  mlflow.set_experiment(cfg.name)

  return mlflow.get_experiment(experiment_id)

# def set_or_create_experiment(name:str, artifact_dir: Path):
#   '''
#   helper to create mlflow experiment
#   '''
#   # TRACKING_URI = f"sqlite:///{Path('experiments/mlflow.db').resolve()}" 
#   # mlflow.set_tracking_uri(TRACKING_URI)
#   # print(f"TRACKING URI : {TRACKING_URI}")
# 
#   experiment = mlflow.get_experiment_by_name(name)
#   if experiment is None:
#     experiment_id = mlflow.create_experiment(name=name, artifact_location=artifact_dir.resolve().as_uri())
#   else:
#     experiment_id = experiment.experiment_id
# 
#   mlflow.set_experiment(name)
# 
#   return mlflow.get_experiment(experiment_id)

'''
def build_weighted_sampler(dataloader:DataLoader):
  class_counts = np.bincount(dataloader.dataset.targets)
  class_weights = 1.0 / class_counts
  sample_weights = class_weights[dataloader.dataset.targets] # [class_weights[t] for t in train_dataset.targets]
  sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

  return sampler
'''