
## General structure

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
