import torch
import torch.nn as nn

class FocalLoss(nn.Module):
  def __init__( self, alpha:torch.Tensor=None, gamma=2.0, reduction='mean'):
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction
    self.ce_fn = nn.CrossEntropyLoss(reduction='none')

  def forward(self, logits, targets):
    ce_loss = self.ce_fn(logits, targets)
    pt = torch.exp(-ce_loss)
    focal_loss = (1 - pt) ** self.gamma * ce_loss
    # print(type(self.alpha), self.alpha.device)

    if self.alpha is not None:
      alpha_t = self.alpha[targets]
      focal_loss = alpha_t * focal_loss

    if self.reduction == 'mean':
      return focal_loss.mean()
    elif self.reduction == 'sum':
      return focal_loss.sum()

    return focal_loss
