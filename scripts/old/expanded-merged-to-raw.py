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

  def to_xyxy_expanded(self, scale:float, img_h, img_w) -> tuple[int, int, int, int]:
    '''
    TODO: add min_size: make cropping boxes at least that big, e.g. 32 pixels
    '''
    xc = self.xc * img_w
    yc = self.yc * img_h
    bw = self.w * img_w
    bh = self.h * img_h

    bw *= scale
    bh *= scale
    # enforce minimum crop size
    # bw = max(bw, min_size)
    # bh = max(bh, min_size)
    x1 = xc - bw / 2
    y1 = yc - bh / 2
    x2 = xc + bw / 2
    y2 = yc + bh / 2
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(img_w, int(round(x2)))
    y2 = min(img_h, int(round(y2)))

    return x1, y1, x2, y2 

BLUEBERRY_ROOT = Path.home() / "BLUEBERRY_DATA"
MERGED_ROOT = BLUEBERRY_ROOT / "MERGED"
LABELS_ROOT =  MERGED_ROOT / "labels"
IMAGES_ROOT =  MERGED_ROOT / "images"

EXPANDED_RAW_DATA_ROOT = Path("data/raw/classification-expanded-boxes")
# CLASSIFICATION_ROOT =  BLUEBERRY_ROOT / "CLASSIFICATION"

# for label_file in islice(LABELS_ROOT.iterdir(), 2):
for label_file in LABELS_ROOT.iterdir():
  img_name = label_file.with_suffix(".png").name 
  # print(img_name, label_file.stem)
  img_path = IMAGES_ROOT / img_name
  # assert img_path.exists()
  img = cv2.imread(img_path)
  img_h, img_w, _ = img.shape
  with label_file.open() as f:
    for i, line in enumerate(f):
      cls, cx, cy, w, h = map(float, line.split())
      cls = int(cls)
      bbox = BBox(cls,cx,cy,w,h)
      x1, y1, x2, y2 = bbox.to_xyxy_expanded(2.0, img_h, img_w)
      # x1, y1, x2, y2 = bbox.to_xyxy(img_h, img_w)

      crop = img[y1:y2, x1:x2]
      # path to write the bbox for classification
      dst_box_path = EXPANDED_RAW_DATA_ROOT / f"{str(cls) + '/' + img_path.stem}_{i}.png"
      # print(dst_box_path)
      cv2.imwrite(dst_box_path, crop)