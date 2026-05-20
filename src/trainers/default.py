import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.utils.metrics import compute_classification_metrics

class Trainer:
  def __init__(self, model:nn.Module, train_loader:DataLoader, dev_loader:DataLoader, viz_loader:DataLoader, loss_fn:nn.Module, optimizer:optim, scheduler:lr_scheduler, device:torch.device):
    self.model = model.to(device)
    self.train_loader = train_loader
    self.dev_loader = dev_loader
    self.viz_loader = viz_loader
    self.device = device
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.scheduler = scheduler
    '''
    self.optimizer = optimizer(model.parameters(), lr=lr)
    # self.optimizer = Adam(model.parameters(), lr=lr)
    '''

  def train_one_epoch(self):
    '''
    only returns the epoch loss now
    '''
    self.model.train()
    train_loss = 0
    # correct = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(self.train_loader, desc="Train loader", leave=False)
    for (Xb, Yb) in pbar:
      Xb, Yb = Xb.to(self.device), Yb.to(self.device)

      self.optimizer.zero_grad(set_to_none=True)
      out = self.model(Xb)
      loss = self.loss_fn(out, Yb)
      loss.backward()
      self.optimizer.step()

      train_loss += loss.detach().item()
      _, pred = torch.max(out, 1)
      all_preds.append(pred.cpu())
      all_targets.append(Yb.cpu())
      # correct += pred.eq(Yb).float().sum().item()

    # TODO: this works for update per epoch schedulers. If I use a per batch one stepping here is wrong !
    if self.scheduler is not None:
      self.scheduler.step()

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    metrics = compute_classification_metrics(all_preds, all_targets)
    # print(all_preds[:3], all_targets[:3])
    # return train_loss / len(self.train_loader) #, correct / len(self.train_loader.dataset)
    return train_loss / len(self.train_loader), metrics

  @torch.no_grad
  def validate_one_epoch(self):
    self.model.eval()
    dev_loss = 0
    # correct = 0
    all_preds = []
    all_targets = []
    pbar = tqdm(self.dev_loader, desc="Dev loader", leave=False)
    for (Xb, Yb) in pbar:
      Xb, Yb = Xb.to(self.device), Yb.to(self.device)

      out = self.model(Xb)
      loss = self.loss_fn(out, Yb)
      dev_loss += loss.detach().item()
      _, pred = torch.max(out, 1)
      all_preds.append(pred.cpu())
      all_targets.append(Yb.cpu())
      # correct += pred.eq(Yb).float().sum().item()

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    metrics = compute_classification_metrics(all_preds, all_targets)
    # return dev_loss / len(self.dev_loader), correct / len(self.dev_loader.dataset)
    return dev_loss / len(self.dev_loader), metrics


  @torch.no_grad()
  def prediction_dynamics(self):
    self.model.eval()
    wrong_images = []
    wrong_targets = []
    wrong_preds = []
    wrong_indexes = []
    pbar = tqdm(self.viz_loader, desc="Visualization loader", leave=False)
    for idx, (Xb, Yb) in enumerate(pbar):
      Xb, Yb = Xb.to(self.device), Yb.to(self.device)
      out = self.model(Xb)
      pred = out.argmax(dim=1)

      wrong_mask = pred != Yb

      if wrong_mask.any():
        # print(len(wrong_mask))
        wrong_images.append(Xb[wrong_mask].cpu())
        wrong_targets.append(Yb[wrong_mask].cpu())
        wrong_preds.append(pred[wrong_mask].cpu())

        # this is the index relativee to viz_loader total size.
        viz_indices = torch.arange(Xb.size(0)) + idx * Xb.size(0)
        viz_indices = viz_indices.to(self.device)
        wrong_indexes.append(viz_indices[wrong_mask].cpu())

      # TODO: keep only max_images ?? 

    if len(wrong_images) == 0:
      return None

    wrong_images = torch.cat(wrong_images)
    wrong_targets = torch.cat(wrong_targets)
    wrong_preds = torch.cat(wrong_preds)
    wrong_indexes = torch.cat(wrong_indexes)

    return wrong_images, wrong_targets, wrong_preds, wrong_indexes
