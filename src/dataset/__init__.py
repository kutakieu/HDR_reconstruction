from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader, Dataset


@dataclass
class DataSample:
    img: Path
    corner: Path


class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        self.loader_args = kwargs["loader_args"]

    def create_dataloader(self):
        return DataLoader(self, **self.loader_args)
