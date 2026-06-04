import optuna
from optuna.trial import TrialState
from src.utils.reproducibility import seed_everything


from omegaconf import OmegaConf
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from src.utils.config import BaseConfig

def objective(trial):
  x = trial.suggest_float("x", -10, 10, log=False)

  cs = ConfigStore.instance()
  cs.store(name="base_config", node=BaseConfig)
  with initialize(version_base=None, config_path="../configs"):
    cfg = compose(
      config_name="blueberries_merged_pink_purple_optuna_search",
    )

  print(OmegaConf.to_yaml(cfg))

  for step in range(10):
    value = (x-2)**2 + (10-step)
    trial.report(value, step)

    if trial.should_prune():
      raise optuna.TrialPruned()

  return (x-2)**2

if __name__ == "__main__":
  sampler = optuna.samplers.TPESampler(seed=0)
  pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

  seed_everything(0)
  study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
  study.optimize(objective, n_trials=1, timeout=10) # stop after timeout seconds, or n_trials
  pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
  complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
  print("Number of finished trials: ", len(study.trials))
  print("Number of pruned trials: ", len(pruned_trials))
  print("Number of complete trials: ", len(complete_trials))
  print(f"Best trial:{study.best_trial}")
  print(f"Best value:{study.best_value}, at {study.best_params}")
