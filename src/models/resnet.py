import torch.nn as nn
from torchvision.models import resnet18, resnet50

def get_resnet18(num_classes:int):
  model = resnet18(weights="IMAGENET1K_V1")
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  return model

def get_resnet50(num_classes:int):
  model = resnet50(weights="IMAGENET1K_V2")
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  return model