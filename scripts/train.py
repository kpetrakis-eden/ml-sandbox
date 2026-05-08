from pathlib import Path
import yaml
import os
# needed when use_determinist_algoriithms is used in CUDA > 10.2, before importing pytorch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from src.utils.reproducibility import seed_everything
from src.models.resnet import get_resnet18
from src.datasets.classification import get_dataloaders
from src.trainers.default import Trainer

with open("configs/config.yaml") as f:
  config = yaml.safe_load(f)
print(config)
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")

seed_everything(config['seed'])
generator = torch.Generator().manual_seed(config['seed'])

train_loader, dev_loader = get_dataloaders(config['data_root'], config['batch_size'], generator)
_, targets = next(iter(train_loader))
# print(targets[:10])

model = get_resnet18(num_classes=config['num_classes'])

trainer = Trainer(model, train_loader, dev_loader, device, lr=config['lr'])

for epoch in range(config['epochs']):
  trainer.train_one_epoch()