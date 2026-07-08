import os
# needed when use_determinist_algoriithms is used in CUDA > 10.2, before importing pytorch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"
import optuna
from optuna.trial import TrialState

from omegaconf import OmegaConf
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore

import mlflow
from tqdm import tqdm
import torch
from src.utils.config import BaseConfig
from src.utils.reproducibility import seed_everything
from src.models.factory import get_model
from src.datasets.classification import DataFactory
from src.trainers.default import Trainer
from src.optimizers.optimizer import get_optimizer, get_scheduler
from src.losses.factory import get_loss_fn
from src.utils.extra import set_or_create_experiment

def objective(trial):
  optim = trial.suggest_categorical("optim", ["adamw", "sgd"])
  if optim == "sgd":
    momentum = trial.suggest_float("momentum", 0.7, 0.99)
  else:
    momentum = 0
  lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
  weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)
  batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
  # batch_size = trial.suggest_categorical("batch_size", [120, 250, 500]) # for balanced , multples of 5
  run_name = "swintiny-weighted-bicubic-aug" + f"_trial-{trial.number}"
  # run_name = "convnexttiny-weighted-bicubic-aug" + f"_trial-{trial.number}"

  cs = ConfigStore.instance()
  cs.store(name="base_config", node=BaseConfig)
  # TODO: override experiment name
  with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
      config_name="blueberries_merged_pink_purple_optuna_search",
      overrides=[
        f"optimizer.name={optim}",
        f"optimizer.momentum={momentum}",
        f"optimizer.lr={lr}",
        f"optimizer.weight_decay={weight_decay}",
        f"data.batch_size={batch_size}",
        f"experiment.run_name={run_name}",
      ]
    )
  tqdm.write(OmegaConf.to_yaml(cfg))

  device = torch.device("cuda:1")
  seed_everything(cfg.seed)
  generator = torch.Generator().manual_seed(cfg.seed)
  data_factory = DataFactory(cfg.data, generator)
  train_loader, dev_loader = data_factory.build_datasets().build_sampler().build_loaders()
  viz_loader = None # no prediction dynamics here
  model = get_model(cfg) # before: get_model(cfg.model)
  loss_fn = get_loss_fn(cfg.loss, train_loader, device)
  optimizer = get_optimizer(cfg.optimizer, model)
  scheduler = get_scheduler(cfg.scheduler, optimizer)
  trainer = Trainer(model, train_loader, dev_loader, viz_loader, loss_fn, optimizer, scheduler, device)

  experiment = set_or_create_experiment(cfg.experiment)
  with mlflow.start_run(run_name=cfg.experiment.run_name) as run:
    mlflow.log_params(cfg)

    best_f1_macro = 0
    pbar = tqdm(range(1, cfg.epochs+1), desc="Main Loop", unit="epoch")
    for epoch in pbar:
      # tqdm.write(f"using lr: {optimizer.param_groups[0]['lr']}") # to verify scheduler works as expected
      loss, metrics = trainer.train_one_epoch()
      dev_loss, dev_metrics = trainer.validate_one_epoch()
      pbar.set_postfix({ "train_loss": f"{loss:.7f}", "dev_loss": f"{dev_loss:.7f}"})
      mlflow.log_metrics({
        "train/loss": loss,
        "train/acc": metrics['acc'],
        "train/balanced_acc": metrics['balanced_acc'],
        "train/f1_macro": metrics['f1_macro'],
        "train/f1_weighted": metrics['f1_weighted'],
        "train/precision_macro": metrics['precision_macro'],
        "train/precision_weighted": metrics['precision_weighted'],
        "train/recall_macro": metrics['recall_macro'],
        "train/recall_weighted": metrics['recall_weighted'],
        "dev/loss": dev_loss,
        "dev/acc": dev_metrics['acc'],
        "dev/balanced_acc": dev_metrics['balanced_acc'],
        "dev/f1_macro": dev_metrics['f1_macro'],
        "dev/f1_weighted": dev_metrics['f1_weighted'],
        "dev/precision_macro": dev_metrics['precision_macro'],
        "dev/precision_weighted": dev_metrics['precision_weighted'],
        "dev/recall_macro": dev_metrics['recall_macro'],
        "dev/recall_weighted": dev_metrics['recall_weighted'],
      }, step=epoch)

      best_f1_macro = max(best_f1_macro, dev_metrics['f1_macro'])

      # trial.report(dev_loss, epoch)
      # trial.report(dev_metrics['f1_macro'], epoch)
      trial.report(best_f1_macro, epoch)
      # Handle pruning based on the intermediate value.
      if trial.should_prune():
          raise optuna.TrialPruned()

  # return dev_loss # should minimize
  # NOTE: those should have maximize
  # return dev_metrics['f1_macro'] # trial measured on last epoch
  return best_f1_macro # trial measured on best epoch

if __name__ == "__main__":
  '''
  TODO: run trials searching for max f1 score, instead of min loss
  '''
  N_TRIALS = 100
  TIMEOUT = 61200 # 216000 # 54000 # sec
  optuna_sampler = optuna.samplers.TPESampler(seed=0)
  # pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5) # TODO: use HyperbandPruner is for DL
  pruner = optuna.pruners.HyperbandPruner(min_resource=2, max_resource=20, reduction_factor=3)
  # study = optuna.create_study(direction="minimize" ,sampler=optuna_sampler, pruner=pruner)
  study = optuna.create_study(direction="maximize" ,sampler=optuna_sampler, pruner=pruner)
  study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)

  pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
  complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

  print("Study statistics: ")
  print("Number of finished trials: ", len(study.trials))
  print("Number of pruned trials: ", len(pruned_trials))
  print("Number of complete trials: ", len(complete_trials))

  print(f"Best trial: {study.best_trial}")
  print(f"Best value:{study.best_value}, at {study.best_params}")

  print("Best params:")
  for k, v in study.best_params.items():
    print(f"{k}: {v}")
