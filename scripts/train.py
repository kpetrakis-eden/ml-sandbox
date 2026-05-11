import os
# needed when use_determinist_algoriithms is used in CUDA > 10.2, before importing pytorch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"
from pathlib import Path
import yaml
import json
from tqdm import tqdm
import torch
from src.utils.reproducibility import seed_everything
from src.models.resnet import get_resnet18
from src.datasets.classification import get_dataloaders
from src.trainers.default import Trainer

with open("configs/config.yaml") as f:
  config = yaml.safe_load(f)
print(f"Config: \n{json.dumps(config, indent=2)}")
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")

seed_everything(config['seed'])
generator = torch.Generator().manual_seed(config['seed'])

train_loader, dev_loader = get_dataloaders(config['data_root'], config['batch_size'], generator)
# _, targets = next(iter(train_loader))
# print(targets[:10])
model = get_resnet18(num_classes=config['num_classes'])
trainer = Trainer(model, train_loader, dev_loader, device, lr=config['lr'])

pbar = tqdm(range(config['epochs']), desc="Main Loop", unit="epoch")
for epoch in pbar:
  loss, acc = trainer.train_one_epoch()
  dev_loss, dev_acc = trainer.validate_one_epoch()
  pbar.set_postfix({
    "train_loss": f"{loss:.7f}",
    "train_acc": f"{100*acc:2.2f}%",
    "dev_loss": f"{dev_loss:.7f}",
    "dev_acc": f"{100*dev_acc:2.2f}%"
  })

  # train_loss=0.2664522, train acc=90.08 %, dev loss=0.2600182, dev acc=90.48 %
  # 1/3 [01:05<02:11, 65.99s/epoch, train_loss=0.3135408, train_acc=88.33%, dev_loss=0.2811642, dev_acc=89.71%
