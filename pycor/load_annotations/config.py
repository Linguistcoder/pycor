from dataclasses import dataclass
from typing import Optional, List
import pathlib


@dataclass
class DatasetConfig:
    name: str
    file: pathlib.Path
    quote: pathlib.Path
    sample_size: Optional[float]
    bias: Optional[int]
    subsample_size: Optional[float]
    sub_bias: Optional[int]


@dataclass
class Configuration:
    datasets: List[DatasetConfig]
