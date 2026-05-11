import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

class Trainer:
  def __init__(self, model:nn.Module, train_loader:DataLoader, dev_loader:DataLoader, device:torch.device, lr:float):
    self.model = model.to(device)
    self.train_loader = train_loader
    self.dev_loader = dev_loader
    self.device = device
    self.loss_fn = nn.CrossEntropyLoss()
    self.optimizer = Adam(model.parameters(), lr=lr)

  def train_one_epoch(self):
    '''
    only returns the epoch loss now
    '''
    self.model.train()
    train_loss = 0
    correct = 0

    pbar = tqdm(self.train_loader, desc="Train dataloader", leave=False)
    # for i, (Xb, Yb) in enumerate(self.train_loader):
    for (Xb, Yb) in pbar:
      Xb, Yb = Xb.to(self.device), Yb.to(self.device)

      self.optimizer.zero_grad(set_to_none=True)
      out = self.model(Xb)
      loss = self.loss_fn(out, Yb)
      loss.backward()
      self.optimizer.step()

      train_loss += loss.detach().item()
      _, pred = torch.max(out, 1)
      correct += pred.eq(Yb).float().sum().item()

    return train_loss / len(self.train_loader), correct / len(self.train_loader.dataset)

  @torch.no_grad
  def validate_one_epoch(self):
    self.model.eval()
    dev_loss = 0
    correct = 0
    pbar = tqdm(self.dev_loader, desc="Dev dataloader", leave=False)
    # for i, (Xb, Yb) in enumerate(self.dev_loader):
    for (Xb, Yb) in pbar:
      Xb, Yb = Xb.to(self.device), Yb.to(self.device)

      out = self.model(Xb)
      loss = self.loss_fn(out, Yb)
      dev_loss += loss.detach().item()
      _, pred = torch.max(out, 1)
      correct += pred.eq(Yb).float().sum().item()

    return dev_loss / len(self.dev_loader), correct / len(self.dev_loader.dataset)

