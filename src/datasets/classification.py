from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as v2

def get_dataloaders(data_dir: Path, batch_size: int, g: torch.Generator):
  # NOTE: if I add random transforms then I need seed_worker for determinism in dataloading
  transforms = v2.Compose([
    v2.Resize((64, 64)),
    v2.ToImage(),
    # v2.CenterCrop(),

    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.4292, 0.5389, 0.3654], std=[0.1860, 0.2121, 0.1966])
    # [0.4291510581970215, 0.5389042496681213, 0.3654465973377228], [0.18595948815345764, 0.21214619278907776, 0.19659456610679626]
  ])

  data_dir = Path(data_dir)
  train_dataset = ImageFolder(data_dir / "train", transform=transforms)
  dev_dataset = ImageFolder(data_dir / "dev", transform=transforms)

  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, generator=g)
  dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=12, generator=g)

  return train_dataloader, dev_dataloader