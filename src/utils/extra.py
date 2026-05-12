# TODO: name this file appropriately
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import mlflow

def compute_class_weights(dataloader:DataLoader, device:torch.device):
  '''
  there are different ways to compute class weights.. Check notebooks/Classification
  '''
  class_counts = np.bincount(dataloader.dataset.targets)
  class_weights = 1.0 / class_counts
  class_weights = class_weights / class_weights.sum() * len(class_weights)
  class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

  return class_weights


def build_weighted_sampler(dataloader:DataLoader):
  class_counts = np.bincount(dataloader.dataset.targets)
  class_weights = 1.0 / class_counts
  sample_weights = class_weights[dataloader.dataset.targets] # [class_weights[t] for t in train_dataset.targets]
  sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

  return sampler


def set_or_create_experiment(name:str, artifact_dir: Path):
  '''
  helper to create mlflow experiment
  '''
  # TRACKING_URI = f"sqlite:///{Path('experiments/mlflow.db').resolve()}" 
  # mlflow.set_tracking_uri(TRACKING_URI)
  # print(f"TRACKING URI : {TRACKING_URI}")

  experiment = mlflow.get_experiment_by_name(name)
  if experiment is None:
    experiment_id = mlflow.create_experiment(name=name, artifact_location=artifact_dir.resolve().as_uri())
  else:
    experiment_id = experiment.experiment_id

  mlflow.set_experiment(name)

  return mlflow.get_experiment(experiment_id)
