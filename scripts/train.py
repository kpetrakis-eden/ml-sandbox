import os
# needed when use_determinist_algoriithms is used in CUDA > 10.2, before importing pytorch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"
from pathlib import Path
import yaml
import json
from tqdm import tqdm
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from src.utils.reproducibility import seed_everything
from src.models.resnet import get_resnet18
from src.datasets.classification import get_dataloaders
from src.trainers.default import Trainer
from src.losses.factory import get_loss_fn
from src.utils.extra import compute_class_weights

with open("configs/config.yaml") as f:
  config = yaml.safe_load(f)
print(f"Config: \n{json.dumps(config, indent=2)}")
CLASS_NAMES = config['class_names']
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")

seed_everything(config['seed'])
generator = torch.Generator().manual_seed(config['seed'])

train_loader, dev_loader = get_dataloaders(config['data_root'], config['batch_size'], generator)
# _, targets = next(iter(train_loader))
# print(targets[:10])
model = get_resnet18(num_classes=config['num_classes'])
# class_weights = compute_class_weights(train_loader, device)
# print(f"Class weights: {class_weights}")
class_weights = None
loss_fn = get_loss_fn(config, class_weights)
optimizer = Adam
trainer = Trainer(model, train_loader, dev_loader, loss_fn, optimizer, device, lr=config['lr'])

pbar = tqdm(range(config['epochs']), desc="Main Loop", unit="epoch")
for epoch in pbar:
  # loss, acc = trainer.train_one_epoch()
  loss, metrics = trainer.train_one_epoch()
  # dev_loss, dev_acc = trainer.validate_one_epoch()
  dev_loss, dev_metrics = trainer.validate_one_epoch()
  pbar.set_postfix({
    "train_loss": f"{loss:.7f}",
    # "train_acc": f"{100*acc:2.2f}%",
    "dev_loss": f"{dev_loss:.7f}",
    # "dev_acc": f"{100*dev_acc:2.2f}%"
  })
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

  conf_matrix = dev_metrics['confusion_matrix']
  disp_conf_matrix = ConfusionMatrixDisplay(conf_matrix, display_labels=CLASS_NAMES)
  # fig, ax = plt.subplots()
  disp_conf_matrix.plot(cmap=plt.cm.Blues, xticks_rotation=45)
  # plt.show()
  plt.tight_layout()
  plt.savefig(f"dev_cm_noweights_{epoch}.png")
  plt.close()
