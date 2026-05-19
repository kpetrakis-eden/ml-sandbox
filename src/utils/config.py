from pathlib import Path
from typing import List, Optional, Any
from dataclasses import dataclass, field
from omegaconf import DictConfig
# LossName: TypeAlias = Literal["cross-entropy", "focal"]

@dataclass
class MLFlowConfig:
  name: str # experiment name 
  run_name: str
  tracking_uri: Path
  artifact_location: Path 

@dataclass
class LossConfig:
  name: str = "cross-entropy"
  focal_gamma: Optional[float] = None
  focal_reduction: Optional[str] = None

@dataclass
class ModelConfig:
  name: str
  pretrained: bool
  num_classes: int
  default_resnet_downsize: bool
  
@dataclass
class NormalizationConfig:
  # enabled: bool = True 
  mean: List[float]
  std: List[float]

'''
@dataclass
class AugmentationConfig:
  pass
'''

@dataclass
class DataConfig:
  root: Path
  num_workers: Optional[int] = 12
  batch_size : int = 256
  sampling: str = "default"
  normalization: Optional[NormalizationConfig] = None
  augmentation: Optional[Any] = None # this is DictConfig

@dataclass
class OptimizerConfig:
  name: str
  lr: float = 1e-4
  weight_decay: float = 0
  betas: Optional[List[float]] = field(default_factory=lambda: [0.9, 0.999])
  momentum: Optional[float] = None
  nesterov: Optional[bool] = False

@dataclass
class SchedulerConfig:
  name: str

@dataclass
class CosineSchedulerConfig(SchedulerConfig):
  name: str = "cosine"
  T_max: int = 10
  eta_min: float = 0.0

@dataclass
class LinearSchedulerConfig(SchedulerConfig):
  name: str = "linear"
  start_factor: float = 1./3
  end_factor: float = 1.0
  total_iters: int = 5

@dataclass
class StepSchedulerConfig(SchedulerConfig):
  name: str = "step"
  step_size: int = 3
  gamma: float = 0.1

@dataclass
class BaseConfig:
  seed: int
  class_names : List[str]
  data: DataConfig
  experiment : MLFlowConfig
  model: ModelConfig
  optimizer: OptimizerConfig
  scheduler: Optional[SchedulerConfig]
  # num_classes: int
  epochs: int
  # data : DataConfig = field(default_factory=DataConfig)
  loss: LossConfig = field(default_factory=LossConfig)
