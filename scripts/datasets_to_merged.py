# Implement in a script what I did in notebooks/Data.ipynb
from pathlib import Path
import shutil
from itertools import islice
from tqdm import tqdm

DRIVE = Path("/mnt/EL_Viewer/kp")
DATA_ROOT = DRIVE / "blueberry_data"
DATASETS_ROOT = DATA_ROOT / "datasets"
print(DATASETS_ROOT.exists())

MERGED_ROOT = DATA_ROOT / "merged"
(MERGED_ROOT / "images").mkdir(parents=True, exist_ok=True)
(MERGED_ROOT / "labels").mkdir(parents=True, exist_ok=True)

# the new way with pink/purple merged, only 5 classes 
GLOBAL_CLASS_DICT = {"bud":0, "buds":0, "Bud":0, 
                     "flower":1, "Flower":1,
                     "green":2, "Green":2, 
                     "pink":3, "pink-purple":3, "purple":3,
                     "blue":4}

# for dataset in islice(DATA_ROOT.iterdir(), 5, 6):
# for dataset in DATA_ROOT.iterdir():
for dataset in DATASETS_ROOT.iterdir():
  # ignore MERGED folder
  # if dataset.name == "MERGED" or "CLASSIFICATION":# or dataset.name == "dataset-c967718b":
  #   continue
  obj_names_path = dataset /"annotations/obj.names"
  local_cls_names = obj_names_path.open(encoding='utf-8').read().splitlines()
  for cls_name in local_cls_names:
    if cls_name not in GLOBAL_CLASS_DICT.keys():
      raise ValueError(f"class {cls_name} not in GLOBAL classes")
  if dataset.name == "dataset-c967718b": # UUID: c967718b-2031-49ca-aa14-080210cc3fc6
    # se auto to dataset ta .txt exoun 0, 2, 3, 4 indexes. Auto giati ipirxe bud (1) alla egine delete.
    # ara thelw [flower, green, pink, blue] na exoun local_idx 0, 2, 3, 4
    local_class_idx_to_global_map = {i: GLOBAL_CLASS_DICT[cls_name] for i, cls_name in zip([0,2,3,4], local_cls_names)}
    print(f"({dataset.name}) Local class names: {local_cls_names} mapped with: {local_class_idx_to_global_map}")
  else:
    local_class_idx_to_global_map = {i: GLOBAL_CLASS_DICT[cls_name] for i, cls_name in enumerate(local_cls_names)}
    print(f"({dataset.name}) Local class names: {local_cls_names} mapped with: {local_class_idx_to_global_map}")

  labels_dir = dataset / "annotations/obj_train_data"
  images_dir = dataset / "images"
  if not labels_dir.exists():
    raise ValueError(f"{labels_dir} not found")

  # ** If i iterate only on the label files, so I ignore all the images with no annotation file. These are not background in most cases**
  # ** check debug below **
  for label_file in labels_dir.iterdir():
    new_lines = []

    with label_file.open(encoding = 'utf-8') as f:
      for line in f:
        parts = line.strip().split()
        local_cls_idx = int(parts[0])
        global_cls_idx = local_class_idx_to_global_map[local_cls_idx]
        parts[0] = str(global_cls_idx)
        new_lines.append(" ".join(parts))
        # print(f"{label_file.name} cls_idx {local_cls_idx} changed to {global_cls_idx}")

      # print(f"{label_file.name}, new_lines: {new_lines}")

    # handle possible different image extensions
    stem = label_file.stem
    src_img = None
    # print(stem)
    for ext in (".png", ".jpg", ".jpeg"):
      candidate = images_dir / f"{stem}{ext}"
      if candidate.exists():
        src_img = candidate
        break

    if src_img is None:
      # raise FileNotFoundError(f"{src_img} does not exist")
      print(f"WARNING: no image found for {label_file.name}, skipping")
      continue

    # write new label file, after src_img check to avoid writing label files that don;t correspond to image
    out_label_path = MERGED_ROOT / "labels" / label_file.name
    out_label_path.write_text("\n".join(new_lines))

    # copy corresponding image
    # print(src_img.name)
    dst_img = MERGED_ROOT / "images" / src_img.name
    shutil.copy2(src_img, dst_img)

    '''
    # copy corresponding image
    img_name = label_file.with_suffix(".png").name
    src_img = images_dir / img_name
    dst_img = MERGED_ROOT / "images" / img_name
    # print(img_name)
    # print(src_img)
    # print(dst_img)

    # if src_img.exists():
      # shutil.copy(src_img ,dst_img)
      # pass
    # else:
      # raise FileNotFoundError(f"{src_img} does not exist")
    '''

  print("===============================================================")