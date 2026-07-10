import math
import operator
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.utils.config import BaseConfig
import torchvision.transforms.v2 as v2
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

def compute_classification_metrics(preds, targets, num_classes):
  """
  preds, targets : 1D arrays of int label ids
  num_classes    : total number of classes in the problem (label set = range(num_classes))
  """
  label_ids = list(range(num_classes))            # global, stable label set
  present_labels = sorted(set(targets))           # classes actually in THIS eval set
  coverage = len(present_labels) / num_classes

  accuracy = accuracy_score(targets, preds)
  balanced_accuracy = balanced_accuracy_score(targets, preds)

  # --- global (comparable across runs/datasets); absent classes count as 0 ---
  f1_macro     = f1_score(targets, preds, labels=label_ids, average='macro',    zero_division=0)
  f1_weighted  = f1_score(targets, preds, labels=label_ids, average='weighted', zero_division=0)
  f1_per_class = f1_score(targets, preds, labels=label_ids, average=None,       zero_division=0)

  # --- present-only (honest number for partial probe datasets) ---
  f1_macro_present = f1_score(targets, preds, labels=present_labels, average='macro', zero_division=0)

  precision_macro    = precision_score(targets, preds, labels=label_ids, average='macro',    zero_division=0)
  precision_weighted = precision_score(targets, preds, labels=label_ids, average='weighted', zero_division=0)
  recall_macro       = recall_score(targets, preds, labels=label_ids, average='macro',       zero_division=0)
  recall_weighted    = recall_score(targets, preds, labels=label_ids, average='weighted',    zero_division=0)

  conf_matrix = confusion_matrix(targets, preds, labels=label_ids, normalize=None)   # <- labels= fixes shape
  per_class_report = classification_report(targets, preds, labels=label_ids, output_dict=True, zero_division=0)

  return {
    "acc": accuracy * 100,
    "balanced_acc": balanced_accuracy * 100,
    "f1_macro": f1_macro * 100,
    "f1_macro_present": f1_macro_present * 100,
    "f1_per_class": {i: 100 * f for i, f in enumerate(f1_per_class)},  # per-class curves
    "f1_weighted": f1_weighted * 100,
    "precision_macro": precision_macro * 100,
    "precision_weighted": precision_weighted * 100,
    "recall_macro": recall_macro * 100,
    "recall_weighted": recall_weighted * 100,
    "confusion_matrix": conf_matrix,
    "classification_report": per_class_report,
    "coverage": coverage * 100,
  }

"""
def compute_classification_metrics(preds, targets):
  '''
  My previous initial implementation, with modifications of LABEL_IDS and present_labels used by finetuning.ipynb
  '''
  LABEL_IDS = [0, 1, 2, 3, 4] # for consistency in finetuning results
  present_labels = sorted(set(targets))
  coverage = len(set(targets)) / len(LABEL_IDS)

  accuracy = accuracy_score(targets, preds)
  balanced_accuracy = balanced_accuracy_score(targets, preds)

  f1_macro = f1_score(targets, preds, labels=LABEL_IDS, average='macro')
  # Given the classes that exist in this dataset, how well did the model learn them?
  f1_macro_present = f1_score(targets, preds, labels=present_labels, average='macro', zero_division=0)
  f1_macro_per_class = f1_score(targets, preds, labels=LABEL_IDS, average=None, zero_division=0)
  f1_weighted = f1_score(targets, preds, labels=LABEL_IDS, average='weighted')

  precision_macro = precision_score(targets, preds, labels=LABEL_IDS, average='macro', zero_division='warn')
  precision_weighted = precision_score(targets, preds, labels=LABEL_IDS, average='weighted', zero_division='warn')

  recall_macro = recall_score(targets, preds, labels=LABEL_IDS, average='macro', zero_division='warn')
  recall_weighted = recall_score(targets, preds, labels=LABEL_IDS, average='weighted', zero_division='warn')

  # conf_matrix = confusion_matrix(targets, preds, normalize='true')
  conf_matrix = confusion_matrix(targets, preds, normalize=None)
  # disp = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names, cmap=plt.cm.Blues)

  per_class_report = classification_report(targets, preds, labels=LABEL_IDS, output_dict=True)

  return {
    "acc": accuracy * 100,
    "balanced_acc": balanced_accuracy * 100,
    "f1_macro": f1_macro * 100,
    "f1_macro_present": f1_macro_present*100,
    # "f1_macro_per_class": [100*f1 for f1 in f1_macro_per_class],
    "f1_weighted": f1_weighted * 100,
    "precision_macro": precision_macro * 100,
    "precision_weighted": precision_weighted * 100,
    "recall_macro": recall_macro * 100,
    "recall_weighted": recall_weighted * 100,
    "confusion_matrix": conf_matrix,
    "classification_report": per_class_report,
    "coverage": coverage * 100
  }
"""

def compute_classification_metrics_for_classifier_yolo_comparison(preds, targets, label_ids):
  '''
  TODO: zero_division might depend on problem setup
  '''
  present_labels = sorted(set(targets))
  accuracy = accuracy_score(targets, preds)
  balanced_accuracy = balanced_accuracy_score(targets, preds)

  # before
  # f1_macro = f1_score(targets, preds, average='macro')
  # f1_weighted = f1_score(targets, preds, average='weighted')
  f1_macro = f1_score(targets, preds, labels=label_ids, average='macro')
  f1_macro_present = f1_score(targets, preds, labels=present_labels, average='macro', zero_division=0)
  f1_weighted = f1_score(targets, preds, labels=label_ids, average='weighted')

  # before
  # precision_macro = precision_score(targets, preds, average='macro', zero_division=0)
  # precision_weighted = precision_score(targets, preds, average='weighted', zero_division=0)
  precision_macro = precision_score(targets, preds, labels=label_ids, average='macro', zero_division=0)
  precision_weighted = precision_score(targets, preds, labels=label_ids, average='weighted', zero_division=0)

  # before
  # recall_macro = recall_score(targets, preds, average='macro', zero_division=0)
  # recall_weighted = recall_score(targets, preds, average='weighted', zero_division=0)
  recall_macro = recall_score(targets, preds, labels=label_ids, average='macro', zero_division=0)
  recall_weighted = recall_score(targets, preds, labels=label_ids, average='weighted', zero_division=0)

  # conf_matrix = confusion_matrix(targets, preds, normalize='true')
  conf_matrix = confusion_matrix(targets, preds, normalize=None, labels=label_ids)
  # disp = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names, cmap=plt.cm.Blues)

  per_class_report = classification_report(targets, preds, output_dict=True, zero_division=0)

  return {
    "acc": accuracy * 100,
    "balanced_acc": balanced_accuracy * 100,
    "f1_macro": f1_macro * 100,
    "f1_macro_present": f1_macro_present*100,
    "f1_weighted": f1_weighted * 100,
    "precision_macro": precision_macro * 100,
    "precision_weighted": precision_weighted * 100,
    "recall_macro": recall_macro * 100,
    "recall_weighted": recall_weighted * 100,
    "confusion_matrix": conf_matrix,
    "classification_report": per_class_report
  }


def plot_pred_dynamics(images:torch.Tensor, targets:torch.Tensor, preds:torch.Tensor, indexes:torch.Tensor, cfg:BaseConfig):
  class_names = cfg.class_names
  data_cfg = cfg.data
  N = len(images)
  cols = 8
  rows = math.ceil(N / cols)

  if data_cfg.normalization is not None:
    mean = list(map(operator.neg, data_cfg.normalization.mean))
    std = list(map(partial(operator.truediv, 1), data_cfg.normalization.std))
    denorm = v2.Compose([
      v2.Normalize(mean=[0,0,0], std=std),
      v2.Normalize(mean=mean, std=[1,1,1])
    ])
  else:
    denorm = v2.Identity()

  fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
  axes = axes.flatten()
  for ax, img, target, pred, index in zip(axes, images, targets, preds, indexes):

    # TODO: check the implications of this clamp!
    img = denorm(img).clamp(0, 1) # clamped to avoid warning
    # img = (img - img.min()) / (img.max() - img.min()) # This artifically enhances the image
    img = img.permute(1,2,0).numpy()
    ax.imshow(img)
    ax.set_title(f"{class_names[pred.item()]} ({class_names[target.item()]})", fontsize=12)
    ax.text(0.5, -0.12, f"{index}", transform=ax.transAxes, ha="center", fontsize=10)
    ax.axis("off")

  for ax in list(axes)[N:]:
    ax.axis("off")

  plt.tight_layout()
  return fig

'''
def prediction_dynamics(targets, preds, cls_idx, max_images=16):
  wrong_mask = preds != targets

  # Restrict to specific true class if requested
  if cls_idx is not None:
    wrong_mask &= (targets == cls_idx)

  wrong_indices = np.where(wrong_mask)[0]

  # Limit number of displayed images
  wrong_indices = wrong_indices[:max_images]
'''