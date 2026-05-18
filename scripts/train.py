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
from src.optimizers.optimizer import get_optimizer
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

  device = torch.device("cuda:0")
  seed_everything(cfg.seed)
  generator = torch.Generator().manual_seed(cfg.seed)
  data_factory = DataFactory(cfg.data, generator)
  train_loader, dev_loader = data_factory.build_datasets().build_sampler().build_loaders()
  viz_loader = data_factory.build_viz_subset(cfg.seed).build_viz_loader()
  # _, targets = next(iter(train_loader))
  # print(targets[:10])
  model = get_model(cfg.model)
  loss_fn = get_loss_fn(cfg.loss, train_loader, device)
  # print(loss_fn.weight)
  # optimizer = Adam
  # trainer = Trainer(model, train_loader, dev_loader, viz_loader, loss_fn, optimizer, device, lr=cfg.lr)
  optimizer = get_optimizer(cfg.optimizer, model)
  scheduler = None
  trainer = Trainer(model, train_loader, dev_loader, viz_loader, loss_fn, optimizer, scheduler, device)

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
    best_dev_loss = float('inf')
    best_conf_matrix = None
    pbar = tqdm(range(1, cfg.epochs+1), desc="Main Loop", unit="epoch")
    for epoch in pbar:
      loss, metrics = trainer.train_one_epoch()
      dev_loss, dev_metrics = trainer.validate_one_epoch()
      pbar.set_postfix({ "train_loss": f"{loss:.7f}", "dev_loss": f"{dev_loss:.7f}"})

      mlflow.log_metrics({
        "train/loss": loss,
        "train/acc": metrics['acc'],
        "train/balanced_acc": metrics['balanced_acc'],
        "train/f1_macro": metrics['f1_macro'],
        "train/f1_weighted": metrics['f1_weighted'],
        "train/precision_macro": metrics['precision_macro'],
        "train/precision_weighted": metrics['precision_weighted'],
        "train/recall_macro": metrics['recall_macro'],
        "train/recall_weighted": metrics['recall_weighted'],
        "dev/loss": dev_loss,
        "dev/acc": dev_metrics['acc'],
        "dev/balanced_acc": dev_metrics['balanced_acc'],
        "dev/f1_macro": dev_metrics['f1_macro'],
        "dev/f1_weighted": dev_metrics['f1_weighted'],
        "dev/precision_macro": dev_metrics['precision_macro'],
        "dev/precision_weighted": dev_metrics['precision_weighted'],
        "dev/recall_macro": dev_metrics['recall_macro'],
        "dev/recall_weighted": dev_metrics['recall_weighted'],
      }, step=epoch)

      if (epoch == 1) or (epoch % 5) == 0:
        conf_matrix = dev_metrics['confusion_matrix']
        fig, ax = plt.subplots(figsize=(6,6))
        disp_conf_matrix = ConfusionMatrixDisplay(conf_matrix, display_labels=cfg.class_names)
        disp_conf_matrix.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
        mlflow.log_figure(fig, f"confusion_matrices/conf_matrix_{epoch:03d}.png")
        plt.close(fig)

        wrong_images, wrong_targets, wrong_preds, wrong_indexes = trainer.prediction_dynamics()
        pred_dynamics_fig = plot_pred_dynamics(wrong_images, wrong_targets, wrong_preds, wrong_indexes, cfg)
        mlflow.log_figure(pred_dynamics_fig, f"prediction_dynamics/{epoch:03d}.png")

      if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # keep metrics at lowest loss
        best_conf_matrix = dev_metrics['confusion_matrix']
        report = dev_metrics['classification_report']

        wrong_images, wrong_targets, wrong_preds, wrong_indexes = trainer.prediction_dynamics()
        best_pred_dynamics_fig = plot_pred_dynamics(wrong_images, wrong_targets, wrong_preds, wrong_indexes, cfg)

    # log best classification report
    with open("experiments/classification_report_at_min_loss.json", "w") as f:
      json.dump(report, f, indent=2)
    mlflow.log_artifact("experiments/classification_report_at_min_loss.json", artifact_path="reports")

    # log confusion matrix at lowest loss
    fig, ax = plt.subplots(figsize=(6,6))
    disp_conf_matrix = ConfusionMatrixDisplay(best_conf_matrix, display_labels=cfg.class_names)
    disp_conf_matrix.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
    mlflow.log_figure(fig, f"confusion_matrices/conf_matrix_at_min_loss.png")
    plt.close(fig)

    # log pred dynamics at lowest loss
    mlflow.log_figure(best_pred_dynamics_fig, f"prediction_dynamics/at_min_loss.png")

    # log model at lowest loss
    model.load_state_dict(best_state)
    mlflow.pytorch.log_model(model, name="best_model")


if __name__ == "__main__":
  main()