from src.utils.config import ModelConfig, BaseConfig
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights

def build_resnet18(cfg: ModelConfig):
  weights = ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
  model = resnet18(weights=weights)

  # remove aggresive initial downsize
  if not cfg.default_resnet_downsize:
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

  model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
  return model


def get_model(cfg:ModelConfig):
  match cfg.name:
    case "resnet18":
      return build_resnet18(cfg)
    case _:
      raise ValueError(f"Not supported model: {cfg.name}")
    # TODO..

'''
def get_resnet18(num_classes:int):
  model = resnet18(weights="IMAGENET1K_V1")
  # TODO: modify model for tiny objects
  # model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
  # model.maxpool = nn.Identity()
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  return model

def get_resnet50(num_classes:int):
  model = resnet50(weights="IMAGENET1K_V2")
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  return model
'''