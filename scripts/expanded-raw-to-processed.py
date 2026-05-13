from itertools import islice
from pathlib import Path
import random
import shutil
from tqdm import tqdm

# valid paths if I run it from root, e.g. uv run scripts/process_classsification_dataset.py
SRC_DATA_ROOT = Path("data/raw/classification-expanded-boxes")
DST_DATA_ROOT = Path("data/processed/classification-expanded-boxes")
assert SRC_DATA_ROOT.exists()
assert DST_DATA_ROOT.exists()

(train_folder := DST_DATA_ROOT / "train").mkdir(parents=True, exist_ok=True)
(dev_folder := DST_DATA_ROOT / "dev").mkdir(parents=True, exist_ok=True)
# print(train_folder, dev_folder)

train_ratio = 0.9 # 90% train, 10% dev
SEED = 42
random.seed(42)

for cls_dir in tqdm(SRC_DATA_ROOT.iterdir()):
  cls_name = cls_dir.name
  # create cls subfolders in train/ dev/
  (train_class_folder := train_folder / cls_name).mkdir(parents=True, exist_ok=True)
  (dev_class_folder := dev_folder / cls_name).mkdir(parents=True, exist_ok=True)

  images = list(cls_dir.glob("*.png"))
  random.shuffle(images)

  split_idx = int(len(images) * train_ratio)
  train_images = images[:split_idx]
  dev_images = images[split_idx:]

  # copy files
  for img_path in train_images:
    shutil.copy(img_path, train_class_folder / img_path.name)
  for img_path in dev_images:
    shutil.copy(img_path, dev_class_folder / img_path.name)