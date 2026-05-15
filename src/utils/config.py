from pathlib import Path
from typing import List, Optional, Literal, TypeAlias
from dataclasses import dataclass, field
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
class NormalizationConfig:
  # enabled: bool = True 
  mean: List[float]
  std: List[float]

@dataclass
class AugmentationConfig:
  pass

@dataclass
class DataConfig:
  root: Path
  num_workers: Optional[int] = 12
  batch_size : int = 256
  sampling: str = "default"
  normalization: Optional[NormalizationConfig] = None
  augmentation: Optional[AugmentationConfig] = None

@dataclass
class BaseConfig:
  seed: int
  class_names : List[str]
  data: DataConfig
  experiment : MLFlowConfig
  # data : DataConfig = field(default_factory=DataConfig)
  loss: LossConfig = field(default_factory=LossConfig)
