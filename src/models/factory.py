from src.utils.config import ModelConfig, BaseConfig
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, efficientnet_v2_s,EfficientNet_V2_S_Weights, vit_b_16, ViT_B_16_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

def build_resnet18(cfg: ModelConfig):
  weights = ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
  model = resnet18(weights=weights)
  # remove aggresive initial downsize
  if not cfg.default_resnet_downsize:
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

  model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
  return model

def build_vit(cfg: ModelConfig):
  pass
  # weights = ViT_B_16_Weights.IMA

def build_efficientnet(cfg:ModelConfig):
  weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if cfg.pretrained else None
  model = efficientnet_v2_s(weights=weights)
  model.classifier[1] = nn.Linear(model.classifier[1].in_features, cfg.num_classes)

  return model

def build_convnext(cfg:ModelConfig):
  weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if cfg.pretrained else None
  model = convnext_tiny(weights=weights)
  model.classifier[2] = nn.Linear(model.classifier[2].in_features, cfg.num_classes)

  return model


def get_model(cfg:ModelConfig):
  match cfg.name:
    case "resnet18":
      return build_resnet18(cfg)
    case "efficientnet":
      return build_efficientnet(cfg)
    case "convnext":
      return build_convnext(cfg)
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