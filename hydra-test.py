import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from dataclasses import dataclass

@dataclass
class Normalization:
    enabled: bool
    mean: list[float]
    std: list[float]

@dataclass
class DataConfig:
    root: str
    num_workers: int
    batch_size: int
    sampling: str
    normalization: Normalization

@dataclass
class Config:
    run_name: str
    seed: int
    num_classes: int
    lr: float
    epochs: int
    loss_name: str
    loss: dict
    data: DataConfig
    class_names: list[str]

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

@hydra.main(version_base=None, config_path="configs", config_name="config_expanded.yaml")
def main(cfg:Config):
  print(type(cfg))
  print(cfg)
  print(type(cfg.data))
  # print(cfg.data)

  print(HydraConfig.get().job.name)


# @hydra.main(version_base=None, config_path="configs", config_name="config_expanded.yaml")
# def main(cfg:DictConfig):
#   # print(OmegaConf.to_yaml(cfg))
#   # print(OmegaConf.to_container(cfg, resolve=True))
#   cfg_dataclass = OmegaConf.to_object(cfg)
#   print(cfg_dataclass)

if __name__ == "__main__":
  main()