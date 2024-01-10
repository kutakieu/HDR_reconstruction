from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader, Dataset


@dataclass
class DataSample:
    hdr_file: Path
    ldr_file: Path


class BaseDataset(Dataset):
    data_samples: list[DataSample]
    
    def __init__(self, **kwargs):
        self.loader_args = kwargs["loader_args"]

    def create_dataloader(self):
        return DataLoader(self, **self.loader_args)
