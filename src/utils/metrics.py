import torch

def accuracy(logits, targets):
  '''
  TODO
  '''
  _, pred = torch.max(logits, 1)
  correct += pred.eq(targets).float().sum().item()

  return 100 * correct # ?

def confusion_matrix():
  raise NotImplemented