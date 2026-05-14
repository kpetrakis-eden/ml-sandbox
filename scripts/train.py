import os
# needed when use_determinist_algoriithms is used in CUDA > 10.2, before importing pytorch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"
import sys
import platform
from pathlib import Path
import yaml
import json
from tqdm import tqdm
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from src.utils.reproducibility import seed_everything
from src.models.resnet import get_resnet18, get_resnet50
from src.datasets.classification import DataConfig, DataFactory # get_dataloaders
from src.trainers.default import Trainer
from src.losses.factory import get_loss_fn
from src.utils.extra import set_or_create_experiment, compute_class_weights, build_weighted_sampler
import mlflow
import logging
import warnings

logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings( "ignore", module="mlflow")

with open("configs/config_expanded.yaml") as f:
  config = yaml.safe_load(f)
print(f"Config: \n{json.dumps(config, indent=2)}")
CLASS_NAMES = config['class_names']
# device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
device = torch.device("cuda:1")
print(device)

seed_everything(config['seed'])
generator = torch.Generator().manual_seed(config['seed'])

data_cfg = DataConfig(**config['data'])
data_factory = DataFactory(data_cfg, generator)
train_loader, dev_loader = data_factory.build_datasets().build_sampler().build_loaders()
# _, targets = next(iter(train_loader))
# print(targets[:10])

model = get_resnet18(num_classes=config['num_classes'])
class_weights = compute_class_weights(train_loader, device)
print(f"Class weights: {class_weights}")
# class_weights = None
loss_fn = get_loss_fn(config, class_weights)
optimizer = Adam
trainer = Trainer(model, train_loader, dev_loader, loss_fn, optimizer, device, lr=config['lr'])

EXPERIMENT_NAME = "blueberries-classification"
# TODO: move those inside helper function
TRACKING_URI = f"sqlite:///{Path('experiments/mlflow.db').resolve()}" 
mlflow.set_tracking_uri(TRACKING_URI)
print(f"TRACKING URI : {TRACKING_URI}")
ARTIFACT_LOCATION = Path("experiments")
experiment = set_or_create_experiment(EXPERIMENT_NAME, ARTIFACT_LOCATION)
print(f"Experiment_id: {experiment.experiment_id}")
print(f"Artifact Location: {experiment.artifact_location}")
print(f"Tags: {experiment.tags}")
print(f"Lifecycle_stage: {experiment.lifecycle_stage}")

RUN_NAME = config["run_name"] # "expanded-resnet18-weighted-sampling"
with mlflow.start_run(run_name=RUN_NAME) as run:
  mlflow.set_tags({
    "stage": "research",
    "model_family": "resnet",
    "environment":"hyperbeast",
    "gpu":"rtx4090",
    "optimizer": "Adam",
    "framework": "PyTorch"
  })
  mlflow.log_params(config)
  mlflow.log_params({
    "python_version": sys.version,
    "pytorch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "platform": platform.platform(),
  })
  best_dev_loss = float('inf')
  best_conf_matrix = None
  pbar = tqdm(range(1, config['epochs']+1), desc="Main Loop", unit="epoch")
  for epoch in pbar:
    loss, metrics = trainer.train_one_epoch()
    dev_loss, dev_metrics = trainer.validate_one_epoch()
    pbar.set_postfix({ "train_loss": f"{loss:.7f}", "dev_loss": f"{dev_loss:.7f}"})
    tqdm.write(
      f"train_acc {metrics['acc']:.2f}% | "
      f"train_balanced_acc: {metrics['balanced_acc']:.2f}% | "
      f"train_f1_macro: {metrics['f1_macro']:.2f}% | "
      f"train_f1_weighted: {metrics['f1_weighted']:.2f}% | "
      f"train_precision_macro: {metrics['precision_macro']:.2f}% | "
      f"train_precision_weighted: {metrics['precision_weighted']:.2f}% | "
      f"train_recall_macro: {metrics['recall_macro']:.2f}% | "
      f"train_recall_weighted: {metrics['recall_weighted']:.2f}%"
    )
    tqdm.write(
      f"dev_acc {dev_metrics['acc']:.2f}% | "
      f"dev_balanced_acc: {dev_metrics['balanced_acc']:.2f}% | "
      f"dev_f1_macro: {dev_metrics['f1_macro']:.2f}% | "
      f"dev_f1_weighted: {dev_metrics['f1_weighted']:.2f}% | "
      f"dev_precision_macro: {dev_metrics['precision_macro']:.2f}% | "
      f"dev_precision_weighted: {dev_metrics['precision_weighted']:.2f}% | "
      f"dev_recall_macro: {dev_metrics['recall_macro']:.2f}% | "
      f"dev_recall_weighted: {dev_metrics['recall_weighted']:.2f}%"
    )
    tqdm.write("="*130 + "\n")
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
    },
    step=epoch)

    if (epoch % 5) == 0:
      conf_matrix = dev_metrics['confusion_matrix']
      fig, ax = plt.subplots(figsize=(6,6))
      disp_conf_matrix = ConfusionMatrixDisplay(conf_matrix, display_labels=CLASS_NAMES)
      disp_conf_matrix.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
      # plt.tight_layout()
      # fig.savefig(f"conf_matrix_{epoch}.png", dpi=300, bbox_inches="tight")
      # mlflow.log_artifact(f"conf_matrix_{epoch}.png", artifact_path="confusion_matrices")
      mlflow.log_figure(fig, f"conf_matrix_{epoch:03d}.png")
      plt.close(fig)

    if dev_loss < best_dev_loss:
      best_dev_loss = dev_loss
      best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
      # torch.save(model.state_dict(), f"best_{epoch:03d}.pt")
      # mlflow.log_artifact("best.pt", model.state_dict())

      # keep metrics at lowest loss
      best_conf_matrix = dev_metrics['confusion_matrix']
      # mlflow.pytorch.log_model(model, name="best_model")

      # full classification report on best loss
      report = dev_metrics['classification_report']

  # log classification report
  with open("experiments/dev_classification_report.json", "w") as f:
    json.dump(report, f, indent=2)
  mlflow.log_artifact("experiments/dev_classification_report.json", artifact_path="reports")

  # model.load_state_dict(best_state)
  # # model.load_state_dict(torch.load("best.pt"))
  # mlflow.pytorch.log_model(model, name="best_model")
  # # mlflow.pytorch.log_model(model, name="best_model", serialization_format="pt2", input_example=torch.randn(1,3,64,64))

  # log best conf matrix
  fig, ax = plt.subplots(figsize=(6,6))
  disp_conf_matrix = ConfusionMatrixDisplay(best_conf_matrix, display_labels=CLASS_NAMES)
  disp_conf_matrix.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
  mlflow.log_figure(fig, f"confusion_matrices/conf_matrix_at_min_loss.png")
  plt.close(fig)

mlflow.end_run()