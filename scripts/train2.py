import os
# needed when use_determinist_algoriithms is used in CUDA > 10.2, before importing pytorch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"
# import sys
# import platform
from pathlib import Path
# import yaml
import json
# from tqdm import tqdm
import torch
# from torch.optim import Adam
# import matplotlib.pyplot as plt
# from sklearn.metrics import ConfusionMatrixDisplay
from src.utils.reproducibility import seed_everything
# from src.models.resnet import get_resnet18, get_resnet50
from src.datasets.classification import DataFactory
# from src.trainers.default import Trainer
from src.losses.factory import get_loss_fn
# from src.utils.extra import set_or_create_experiment, compute_class_weights, build_weighted_sampler
# import mlflow

from src.utils.config import BaseConfig
import hydra
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="base_config", node=BaseConfig)

@hydra.main(version_base=None, config_path="../configs", config_name="config_hydra.yaml")
def main(cfg:BaseConfig):
  print(cfg)

  device = torch.device("cuda:1")
  seed_everything(cfg.seed)
  generator = torch.Generator().manual_seed(cfg.seed)
  data_factory = DataFactory(cfg.data, generator)
  train_loader, dev_loader = data_factory.build_datasets().build_sampler().build_loaders()
  # _, targets = next(iter(train_loader))
  # print(targets[:10])
  loss_fn = get_loss_fn(cfg.loss, train_loader, device)
  # print(loss_fn.weight)


if __name__ == "__main__":
  main()