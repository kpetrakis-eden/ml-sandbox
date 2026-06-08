# THIS WEEK

## Data preprocessing

> __NOTE:__ The below F1 score results correspond to balanced sampling, initial 6-class split and low epochs training runs. They are just indicative of the minor changes preprocessing had.

Initially I resized images to 64x64 using the default interpolation (BILINEAR) 

- To iterate fast I picked a convNext tiny architecture (with balanced sampling) and tried out the following:  

```python
InterpolationMode.NEAREST
InterpolationMode.BICUBIC
InterpolationMode.NEAREST
```
Bicubic seemed to work slightly better (70.6% instaed of 70.0% F1 score).

- I also tried a `letterbox` like resizing of images, where I keep the original image aspect ratio by padding on the smallest side. Available padding modes are: 

```python
constant
edge
reflect
symmetric
```

I only tried training with `edge`, which seemed more natural. This gave a bit more on the above bicubic setup (71.1% compared with 70.6%).

- Also I tried resizing images to 96x96 instead of 64x64. On the above bicubic convnext architecture this gave 71.83% compared with 70.6%.

But I don't know whether this is worth exploring. The 96x96 seemed a bit noisy to me. Perhaps if we add a a bit of padding around initial BBox this would make more sense.

- I should play around a bit more with those..

## HPO

I mainly experimented with the following parameters on augmented and no-augmented datasets.

```python
lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)
batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
```

I use a per model HPO approach. Meaning I keep the model architecture constant (e.g. Resnet18) and I run a sweep to find the best hyperparameters for that model.

### Bugs
- I had a bug in my initial sweep script, I pruned runs checking the F1-macro score (that was correct), but I optimized dev_loss in the wrong direction.
- Also the optimization process only checked the metric at the last epoch. 

### WHAT NOW
- After fixing the above bugs, now I prune and optimize the F1-macro score.  
- Also I modified the objective, so as to keep the best F1 score during the epoch, instead of the F1 score in the last one. 
  - There is some nuance here.. 
  - With this change the more promising runs that will be prioritized in sampling are those that had a higher spike (but might be more noisy, with ups and downs)
  - Keeping final epoch on the other hand is prone to a bad final epoch. (although this is probably preferable for small lr and SGD, which converges slower) 


### CONCLUSIONS
- I plan on running some more HPO sweeps (with the correct objectives/prunes this time) on convNext's (and maybe one other architecture)
- But realistically, at the current time, I think the classification sealing is around 83-84 % F1 score on the 5-class dataseet. 

## TODO
- [ ] Run 2-3 more sweeps until the weekend 
- [ ] Choose the best Hyper Parameters from those and train a classifier on the 5-class Datasets. 
- [ ] Incorporate each complete dataset as it comes into training trials?
  - this would complicate an already messy comparison
  - I was planning to w8 for the next 4-5 datasets to be completed and then re-train on this updated Data version.
- [ ] Deliver 2 models (1 ConvNext and 1 Transformer) to the ML Dashboard so that annotators can choose betwen them and give us feedback on any differences on the field.
