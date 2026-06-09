## Augmentations

I tested the following augmentations (mainly) on Transformers and ConvNext architectures:

- augorig
```yaml
- _target_: torchvision.transforms.v2.RandomHorizontalFlip
  p: 0.5
- _target_: torchvision.transforms.v2.RandomVerticalFlip
  p: 0.5
- _target_: torchvision.transforms.v2.RandomRotation
  degrees: 10
- _target_: torchvision.transforms.v2.ColorJitter
  brightness: 0.05
  contrast: 0.05
  saturation: 0.01
```
- aug
```yaml
- _target_: torchvision.transforms.v2.RandomHorizontalFlip
  p: 0.5
- _target_: torchvision.transforms.v2.RandomVerticalFlip
  p: 0.5
- _target_: torchvision.transforms.v2.RandomRotation
  degrees: 10
- _target_: torchvision.transforms.v2.ColorJitter
  brightness: 0.1
```
- aug2
```yaml
- _target_: torchvision.transforms.v2.RandomHorizontalFlip
  p: 0.5
- _target_: torchvision.transforms.v2.RandomVerticalFlip
  p: 0.5
- _target_: torchvision.transforms.v2.RandomRotation
  degrees: 10
```
- augorig
```yaml
- _target_: torchvision.transforms.v2.RandomHorizontalFlip
  p: 0.5
- _target_: torchvision.transforms.v2.RandomVerticalFlip
  p: 0.5
- _target_: torchvision.transforms.v2.RandomRotation
  degrees: 10
- _target_: torchvision.transforms.v2.ColorJitter
  brightness: 0.1
- _target_: torchvision.transforms.v2.RandomAdjustSharpness
  p: 0.5
  sharpness_factor: 2 
- _target_: torchvision.transforms.v2.GaussianBlur
  kernel_size: 3
  sigma: [0.1, 1]
```

- The one that seems to achieve highest f1-score: `aug` (then `augorig`-`aug3`, `aug2`)

## HPO results

```python
optim = trial.suggest_categorical("optim", ["adamw", "sgd"])
momentum = trial.suggest_float("momentum", 0.7, 0.99)
lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)
batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
```

- ConvNext

I Run ~70 trials throughout the weekend on the above hyperparameters using `aug` augmentation schema from above. It resulted in:

```bash
Best params:
optim: adamw
lr: 1.6762279028107836e-05
weight_decay: 1.1343355851164134e-06
batch_size: 64
```

I Trained this model for 20 epochs (slow due to small batch_size !) and got:

| Val Metric         | result  | 
| :------            | :-----: |
| loss               | 0.27909 |
| acc                | 91.6532 % |
| balanced_acc       | 81.6507 % |
| f1_macro           | 82.6623 % |
| f1_weighted        | 91.6719 % |
| precision_macro    | 81.7711 % |
| precision_weighted | 91.7321 % |
| recall_macro       | 81.6507 % |
| recall_weighted    | 91.6532 % |

Perhaps training a bit more would help!


- Transformers

I run ~ 15 trials yesterday, again using `aug` and got max f1-score at best trial 81.7331 at: 
```bash
Best params:
optim: adamw
lr: 6.75031252159592e-05 
weight_decay: 9.518823489861116e-06
batch_size: 64 
```

## Majority classes mistakes

I trained 1 Swin-tiny Transformer model using `default` instead of `weighted` sampling.
- Weighted sampling is better for imbalance / focus on minority classes
- default just samples the dataset as is

The idea is that when the annotator sees a lot of mispredictions on majority classes (e.g. greens, flowers) he can use this classifier and perhaps get a better result..


## Others
- I also had a look at DVC (lakeFS) to see if there is a better way to organize my datasets into versions and better track model-dataset combos.

Now I have : `classification`, `classification-expanded-boxes` `classification-merged-pink-purple` 

## TODO
- [ ] Sent these models to production
- [ ] Find out how many classes the classifier needs for finetuning
  - this is a bigger task and requires some thinking 