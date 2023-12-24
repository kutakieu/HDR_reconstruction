from typing import List, Literal

from torch.utils.data import random_split

from reconsthdr import Env
from reconsthdr.dataset import BaseDataset, DataSample
from reconsthdr.dataset.hdr import PanoHdrDataset
from reconsthdr.utils.logger import get_logger

logger = get_logger(__name__)

Mode = Literal["tran", "val", "test"]


class DatasetFactory:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        train_subset, val_subset, test_subset = self._split_data_sources(cfg["dataset"]["sources"])
        self.mode_name_to_subset = {
            "train": train_subset,
            "val": val_subset,
            "test": test_subset,
        }

    def create_dataset(self, mode: Mode) -> BaseDataset:
        dataset_subset = self.mode_name_to_subset[mode]
        return PanoHdrDataset(dataset_subset, **self.cfg["dataset"][mode], **self.cfg["dataset"][mode]["aug"])
        
    def _split_data_sources(self, cfg_data_sources):
        train_subset = val_subset = test_subset = None
        for data_source_type in cfg_data_sources.keys():
            valid_sampels = self._extract_valid_samples(cfg_data_sources[data_source_type]["folder_name"])
            cur_train_subset, cur_val_subset, cur_test_subset = random_split(
                valid_sampels,
                [
                    cfg_data_sources[data_source_type]["train"],
                    cfg_data_sources[data_source_type]["val"],
                    cfg_data_sources[data_source_type]["test"],
                ],
            )
            train_subset = cur_train_subset if train_subset is None else (train_subset + cur_train_subset)
            val_subset = cur_val_subset if val_subset is None else (val_subset + cur_val_subset)
            test_subset = cur_test_subset if test_subset is None else (test_subset + cur_test_subset)
        return train_subset, val_subset, test_subset

    def _extract_valid_samples(self, folder_name: str) -> List[DataSample]:
        hdr_folder = Env().data_dir / folder_name / Env().hdr_dir
        ldr_folder = Env().data_dir / folder_name / Env().ldr_dir
        hdr_files = list(hdr_folder.glob("*.hdr"))
        valid_samples: List[DataSample] = []
        for hdr_file in hdr_files:
            ldr_file = ldr_folder / f"{hdr_file.stem}.png"
            if not ldr_file.exists():
                logger.warning(f"hdr file {hdr_file} does not exist")
                continue
            valid_samples.append(DataSample(hdr_file, ldr_file))
        print('num samples:', len(valid_samples))
        return valid_samples
