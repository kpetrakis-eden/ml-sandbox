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

## HPO

### Toy trials on Resnet18

- Me MedianPruner(n_startup_trials=5, n_warmup_steps=5)
```bash
Number of finished trials:  4
Number of pruned trials:  0
Number of complete trials:  4
Best trial:FrozenTrial(number=2, state=<TrialState.COMPLETE: 1>, values=[0.31746160431374393], datetime_start=datetime.datetime(2026, 6, 2, 13, 16, 11, 360831), datetime_complete=datetime.datetime(2026, 6, 2, 13, 44, 28, 559953), params={'optim': 'sgd', 'lr': 0.00036000911929116066, 'weight_decay': 0.0016525280493952577, 'batch_size': 64}, user_attrs={}, system_attrs={}, intermediate_values={1: 0.42620759030550875, 2: 0.3880365108081677, 3: 0.3757102519988574, 4: 0.3545727060442197, 5: 0.3476833436840877, 6: 0.33313518755872784, 7: 0.3310654900732094, 8: 0.32506107627144964, 9: 0.3219148548655579, 10: 0.31746160431374393}, distributions={'optim': CategoricalDistribution(choices=('adamw', 'sgd')), 'lr': FloatDistribution(high=0.001, log=True, low=1e-05, step=None), 'weight_decay': FloatDistribution(high=0.005, log=True, low=1e-06, step=None), 'batch_size': CategoricalDistribution(choices=(64, 128, 256, 512))}, trial_id=2, value=None)
Best value:0.31746160431374393, at {'optim': 'sgd', 'lr': 0.00036000911929116066, 'weight_decay': 0.0016525280493952577, 'batch_size': 64}
Best params:
optim: sgd
lr: 0.00036000911929116066
weight_decay: 0.0016525280493952577
batch_size: 64
```

- Me pruner = optuna.pruners.HyperbandPruner(min_resource=2, max_resource=10, reduction_factor=3)
```bash
Number of finished trials:  7
Number of pruned trials:  4
Number of complete trials:  3
Best trial:FrozenTrial(number=0, state=<TrialState.COMPLETE: 1>, values=[0.4899330057441166], datetime_start=datetime.datetime(2026, 6, 2, 15, 0, 23, 713324), datetime_complete=datetime.datetime(2026, 6, 2, 15, 10, 49, 160517), params={'optim': 'sgd', 'lr': 0.00016051911333587616, 'weight_decay': 0.00010363502339348037, 'batch_size': 512}, user_attrs={}, system_attrs={'completed_rung_0': 61.865706508507955}, intermediate_values={1: 45.1988946971688, 2: 52.29578518966456, 3: 55.63077932600316, 4: 58.44468718041935, 5: 59.911876412400765, 6: 61.865706508507955, 7: 62.83042219996224, 8: 63.61929219182121, 9: 64.45744270496428, 10: 64.54321256778877}, distributions={'optim': CategoricalDistribution(choices=('adamw', 'sgd')), 'lr': FloatDistribution(high=0.001, log=True, low=1e-05, step=None), 'weight_decay': FloatDistribution(high=0.005, log=True, low=1e-06, step=None), 'batch_size': CategoricalDistribution(choices=(64, 128, 256, 512))}, trial_id=0, value=None)
Best value:0.4899330057441166, at {'optim': 'sgd', 'lr': 0.00016051911333587616, 'weight_decay': 0.00010363502339348037, 'batch_size': 512}
Best params:
optim: sgd
lr: 0.00016051911333587616
weight_decay: 0.00010363502339348037
batch_size: 512
```

- Sto hpo.convnexttiny.aug eixa ta akoloutha augmentations:

```yaml
- _target_: torchvision.transforms.v2.RandomHorizontalFlip
  p: 0.5
- _target_: torchvision.transforms.v2.RandomVerticalFlip
  p: 0.5
- _target_: torchvision.transforms.v2.RandomRotation
  degrees: 10
- _target_: torchvision.transforms.v2.ColorJitter
  brightness: 0.1 # 0.05
```

## YOLO Augmentations NOTES

- Mosaic transform might hurt performance
- Mixup blends images -> try mixup 0.0
- reduce scale (0.1 or 0.2) to avoid scaling down the image too much and loose tiny objects
- decrease hsv_s and hsv_v (e.g. 0.4 and 0.2 respectively)
- try:
  - copy_paste 0.0
  - keep perspective, shear very small or 0.0, because blueberries are usualy circle


## General project structure

```bash
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

```