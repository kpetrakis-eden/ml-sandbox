import torch
import torch.nn as nn
from torchvision.models import resnet18

def get_resnet18(num_classes:int):
  model = resnet18(weights="IMAGENET1K_V1")
  # model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
  # model.maxpool = nn.Identity()
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  return model

model = get_resnet18(6)
print(model)
print(model.conv1(torch.randn(1,3,64,64)).size())

# (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# (relu): ReLU(inplace=True)
# (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
