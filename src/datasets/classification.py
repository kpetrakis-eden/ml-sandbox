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
    v2.Normalize(mean=[0.4291, 0.5388, 0.3654], std=[0.1859, 0.2121, 0.1966])
    # [0.42910709977149963, 0.5388360023498535, 0.36541175842285156] , std: [0.1859438121318817, 0.21211254596710205, 0.19656462967395782]
  ])

  data_dir = Path(data_dir)
  train_dataset = ImageFolder(data_dir / "train", transform=transforms)
  dev_dataset = ImageFolder(data_dir / "dev", transform=transforms)

  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, generator=g)
  dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=12, generator=g)

  return train_dataloader, dev_dataloader

'''
v2.Normalize(mean=[0.4292, 0.5389, 0.3654], std=[0.1860, 0.2121, 0.1966])
[0.4291510581970215, 0.5389042496681213, 0.3654465973377228], [0.18595948815345764, 0.21214619278907776, 0.19659456610679626]

Those weights were computed on all the data,before the split..
** THEY MUST NOT BE USED FOR training/dev after splitting.. I have to calculate only the training ones!!

the new values after the split, are not so different!!
'''