from src.utils.config import ModelConfig, BaseConfig
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, efficientnet_v2_s,EfficientNet_V2_S_Weights, vit_b_16, ViT_B_16_Weights
from torchvision.models import convnext_tiny, convnext_base, ConvNeXt_Tiny_Weights, ConvNeXt_Base_Weights
import timm

def build_resnet18(cfg: ModelConfig):
  weights = ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained else None
  model = resnet18(weights=weights)
  # remove aggresive initial downsize
  if not cfg.default_resnet_downsize:
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

  model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
  return model

def build_swin_tiny_patch4(cfg: BaseConfig):
  '''
  TODO: fix the hard-coded image size here (DONE?)
  '''
  model_cfg:ModelConfig = cfg.model
  model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=model_cfg.num_classes, img_size=cfg.data.preprocessing.img_size)
  return model
  
def build_swin_v2_tiny_window8(cfg: BaseConfig):
  model_cfg:ModelConfig = cfg.model
  model = timm.create_model("swinv2_tiny_window8_256", pretrained=True, num_classes=model_cfg.num_classes, img_size=cfg.data.preprocessing.img_size)
  return model

def build_swin_s3_tiny(cfg: BaseConfig):
  model_cfg:ModelConfig = cfg.model
  model = timm.create_model("swin_s3_tiny_224", pretrained=True, num_classes=model_cfg.num_classes, img_size=cfg.data.preprocessing.img_size)
  return model

def build_efficientnet(cfg:ModelConfig):
  weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if cfg.pretrained else None
  model = efficientnet_v2_s(weights=weights)
  model.classifier[1] = nn.Linear(model.classifier[1].in_features, cfg.num_classes)
  return model

def build_convnext_tiny(cfg:ModelConfig):
  weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if cfg.pretrained else None
  model = convnext_tiny(weights=weights)
  model.classifier[2] = nn.Linear(model.classifier[2].in_features, cfg.num_classes)
  return model

def build_convnext_base(cfg:ModelConfig):
  weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if cfg.pretrained else None
  model = convnext_base(weights=weights)
  model.classifier[2] = nn.Linear(model.classifier[2].in_features, cfg.num_classes)
  return model


def get_model(cfg: BaseConfig):
  '''
  NOTE: pass full config in Transformers from timm, to be able to get img_size
  '''
  model_cfg:ModelConfig = cfg.model
  match model_cfg.name:
    case "resnet18":
      return build_resnet18(model_cfg)
    case "efficientnet":
      return build_efficientnet(model_cfg)
    case "convnext-tiny":
      return build_convnext_tiny(model_cfg)
    case "convnext-base":
      return build_convnext_base(model_cfg)
    case "swin_tiny_patch4":
      return build_swin_tiny_patch4(cfg)
    case "swinv2-tiny":
      return build_swin_v2_tiny_window8(cfg)
    case "swin-s3-tiny":
      return build_swin_s3_tiny(cfg)
    case _:
      raise ValueError(f"Not supported model: {model_cfg.name}")
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