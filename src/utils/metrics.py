import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
  accuracy_score,
  balanced_accuracy_score,
  f1_score,
  precision_score,
  recall_score,
  confusion_matrix,
  ConfusionMatrixDisplay
)

def compute_classification_metrics(preds, targets):
  accuracy = accuracy_score(targets, preds)
  balanced_accuracy = balanced_accuracy_score(targets, preds)

  f1_macro = f1_score(targets, preds, average='macro')
  f1_weighted = f1_score(targets, preds, average='weighted')

  precision_macro = precision_score(targets, preds, average='macro', zero_division='warn')
  precision_weighted = precision_score(targets, preds, average='weighted', zero_division='warn')

  recall_macro = recall_score(targets, preds, average='macro', zero_division='warn')
  recall_weighted = recall_score(targets, preds, average='weighted', zero_division='warn')

  conf_matrix = confusion_matrix(targets, preds, normalize='true')
  # disp = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names, cmap=plt.cm.Blues)

  return {
    "acc": accuracy * 100,
    "balanced_acc": balanced_accuracy * 100,
    "f1_macro": f1_macro * 100,
    "f1_weighted": f1_weighted * 100,
    "precision_macro": precision_macro * 100,
    "precision_weighted": precision_weighted * 100,
    "recall_macro": recall_macro * 100,
    "recall_weighted": recall_weighted * 100,
    "confusion_matrix": conf_matrix,
  }

# def accuracy(logits, targets):
#   '''
#   TODO
#   '''
#   _, pred = torch.max(logits, 1)
#   correct += pred.eq(targets).float().sum().item()
# 
#   return 100 * correct # ?
# 
# def confusion_matrix():
#   raise NotImplementedError