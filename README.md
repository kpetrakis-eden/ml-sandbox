# ML Sandbox

## Normalization statistics per case

- cropped boxes resized to 64x64 (default interpolation: BILINEAR):
  - mean: [0.4291, 0.5388, 0.3654]
  - std: [0.1859, 0.2121, 0.1966]

- cropped boxes resized to 64x64 (interpolation: BICUBIC):
  - mean: [0.4291, 0.5388, 0.3654]
  - std: [0.1871, 0.2131, 0.1977]

- cropped boxes resized to 64x64 (interpolation: BICUBIC, PadToSquare('edge')):
  - mean: [0.4276, 0.5372, 0.3643]
  - std: [0.1868, 0.2130, 0.1981]

- cropped boxes resized to 96x96 (interpolation: BICUBIC):
  - mean: [0.4291, 0.5388, 0.3654]
  - std: [0.1871, 0.2131, 0.1978]

- expanded(x2) cropped boxes resized to 64x64 (default interpolation BILINEAR):
  - mean: [0.3994, 0.5100, 0.3541]
  - std: [0.1831, 0.2149, 0.2105]


## General project structure

ml-sandbox/
│
├── data/                  # Raw & processed datasets
│   ├── raw/
│   └── processed/
│
├── configs/               # YAML/JSON configs
│   ├── train.yaml
│   ├── model.yaml
│   └── dataset.yaml
│
├── src/                   # Core ML code
│   ├── datasets/          # Data loading & augmentation
│   │   └── imagenet.py
│   ├── models/            # Model architectures
│   │   └── resnet.py
│   ├── trainers/          # Training & validation loops
│   │   └── pytorch_trainer.py
│   ├── callbacks/         # Custom callbacks (e.g., early stopping)
│   └── utils/             # Logging, metrics, device helpers
│
├── experiments/           # Each experiment gets its own folder
│   ├── exp1/
│   │   ├── checkpoints/
│   │   └── logs/
│   └── exp2/
│
├── logs/                  # TensorBoard, WandB, or MLflow logs
├── scripts/               # CLI entry points
│   ├── train.py
│   └── evaluate.py
├── tests/                 # Unit and integration tests
├── yolo/                  # YOLO specific
├── README.md
