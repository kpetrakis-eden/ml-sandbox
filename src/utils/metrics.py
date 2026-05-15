import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
  accuracy_score,
  balanced_accuracy_score,
  f1_score,
  precision_score,
  recall_score,
  confusion_matrix,
  classification_report,
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

  # conf_matrix = confusion_matrix(targets, preds, normalize='true')
  conf_matrix = confusion_matrix(targets, preds, normalize=None)
  # disp = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names, cmap=plt.cm.Blues)

  per_class_report = classification_report(targets, preds, output_dict=True)

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
    "classification_report": per_class_report
  }


def prediction_dynamics(targets, preds, cls_idx, max_images=16):
  wrong_mask = preds != targets

  # Restrict to specific true class if requested
  if cls_idx is not None:
    wrong_mask &= (targets == cls_idx)

  wrong_indices = np.where(wrong_mask)[0]

  # Limit number of displayed images
  wrong_indices = wrong_indices[:max_images]
