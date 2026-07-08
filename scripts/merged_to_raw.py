from pathlib import Path
from itertools import islice
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
# from torchvision.io import decode_image
from tqdm import tqdm

@dataclass
class BBox:
  cls: int # index
  # cls_name: str
  xc: float # normalized
  yc: float
  w: float
  h: float

  def to_xyxy(self, img_h, img_w) -> tuple[int, int, int, int]:
    x1 = (self.xc - self.w/2) * img_w # x_min
    y1 = (self.yc - self.h/2) * img_h # y_min
    x2 = (self.xc + self.w/2) * img_w # x_max
    y2 = (self.yc + self.h/2) * img_h # y_max
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

DRIVE = Path("/mnt/EL_Viewer/kp")
BLUEBERRY_ROOT = DRIVE / "blueberry_data"
MERGED_ROOT = BLUEBERRY_ROOT / "merged"
LABELS_ROOT =  MERGED_ROOT / "labels"
IMAGES_ROOT =  MERGED_ROOT / "images"

DATASET_NAME = "classification-merged-pink-purple-v2"
RAW_DATA_ROOT = DRIVE / f"data/raw/{DATASET_NAME}"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
image_map = {
  img.stem: img
  for img in IMAGES_ROOT.iterdir()
  if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS
}

# for label_file in islice(LABELS_ROOT.iterdir(), 2):
for label_file in LABELS_ROOT.iterdir():
  img_path = image_map.get(label_file.stem)
  if img_path is None:
    print(f"Missing image for {label_file.name}, skipping")
    continue

  img = cv2.imread(str(img_path))
  img_h, img_w, _ = img.shape
  with label_file.open() as f:
    for i, line in enumerate(f):
      cls, cx, cy, w, h = map(float, line.split())
      cls = int(cls)
      bbox = BBox(cls,cx,cy,w,h)
      x1, y1, x2, y2 = bbox.to_xyxy(img_h, img_w)
      crop = img[y1:y2, x1:x2]
      # path to write the bbox for classification
      # dst_box_path = RAW_DATA_ROOT / f"{str(cls) + '/' + img_path.stem}_{i}.png"
      dst_box_path = RAW_DATA_ROOT / str(cls) / f"{img_path.stem}_{i}.png"
      dst_box_path.parent.mkdir(parents=True, exist_ok=True) # create class folder if it doesn't exist
      cv2.imwrite(dst_box_path, crop)
