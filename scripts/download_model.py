'''
from pathlib import Path
import mlflow
import mlflow.pytorch


if __name__ == "__main__":

  tracking_uri = Path("experiments/mlflow.db").resolve()
  mlflow.set_tracking_uri(f"sqlite:///{tracking_uri}")
  # print(mlflow.get_tracking_uri())
  # run_id = "ff9060a0931e4503b36c090183822424"

  # model = mlflow.pytorch.load_model(f"runs:/{run_id}/best_model")
  # model.eval()
  # print(model)

  experiment = mlflow.get_experiment_by_name("BLUEBERRIES-EXPANDED-BBOXES")
  runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id]
    # order_by=["metrics.dev/loss ASC"]
  )
  print(runs)
'''


#  CHECK model logging/registering..

import os
# needed when use_determinist_algoriithms is used in CUDA > 10.2, before importing pytorch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"
import sys
import platform
from pathlib import Path
import json
from tqdm import tqdm
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from src.utils.reproducibility import seed_everything
from src.models.factory import get_model
from src.datasets.classification import DataFactory
from src.trainers.default import Trainer
from src.optimizers.optimizer import get_optimizer, get_scheduler
from src.losses.factory import get_loss_fn
from src.utils.extra import set_or_create_experiment
from src.utils.metrics import plot_pred_dynamics
import mlflow

from src.utils.config import BaseConfig
import hydra
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="base_config", node=BaseConfig)

@hydra.main(version_base=None, config_path="../configs", config_name="config_hydra.yaml")
def main(cfg:BaseConfig):
  print(cfg)

  seed_everything(cfg.seed)
  generator = torch.Generator().manual_seed(cfg.seed)
  model = get_model(cfg.model)

  experiment = set_or_create_experiment(cfg.experiment)
  with mlflow.start_run(run_name=cfg.experiment.run_name) as run:
    mlflow.set_tags({
      "stage": "research",
      "environment":"hyperbeast",
      "gpu":"rtx4090",
      "framework": "PyTorch"
    })
    mlflow.log_params(cfg)
    mlflow.log_params({
      "python_version": sys.version,
      "pytorch_version": torch.__version__,
      "cuda_version": torch.version.cuda,
      "platform": platform.platform(),
    })


