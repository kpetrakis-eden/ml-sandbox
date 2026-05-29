from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as v2
from src.utils.extra import PadToSquare
from tqdm.auto import tqdm

def compute_normalization_statistics(data_dir:Path, img_size:int, interpolation:v2.InterpolationMode):
  '''
  Calculate train data mean and std for normalization.
  '''
  g = torch.Generator().manual_seed(0)
  transforms = v2.Compose([
    v2.ToImage(),
    # PadToSquare(padding_mode='edge'),
    v2.Resize((img_size, img_size), interpolation=interpolation, antialias=True),
    v2.ToDtype(torch.float32, scale=True)
  ])
  train_dataset = ImageFolder(data_dir / "train", transform=transforms)
  train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=12, generator=g)

  channels_sum = torch.zeros(3, dtype=torch.float64) # 0
  channels_squared_sum = torch.zeros(3, dtype=torch.float64) # 0
  num_pixels = 0

  for images, _ in tqdm(train_loader):
    channels_sum += torch.sum(images, dim=[0, 2, 3])
    channels_squared_sum += torch.sum(images ** 2, dim=[0, 2, 3])

    num_pixels += ( images.shape[0] * images.shape[2] * images.shape[3])

  mean = channels_sum / num_pixels
  std = (channels_squared_sum / num_pixels - mean ** 2) ** 0.5

  return mean, std

# Usage: uv run python -m scripts.normalization_statistics
if __name__ == "__main__":
  # torch.set_printoptions(precision=4)
  IMG_SIZE = 96
  INTERPOLATION = v2.InterpolationMode.BICUBIC
  # data_dir = Path("data/processed/classification-expanded-boxes") # same as in config.yaml
  data_dir = Path("data/processed/classification")
  mean, std = compute_normalization_statistics(data_dir, IMG_SIZE, INTERPOLATION)
  print(f"Mean: {mean} | std: {std}")
  # print(f"Mean: {mean.tolist()} | std: {std.tolist()}")