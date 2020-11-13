from typing import Any, Union, List, Optional

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

import os

from dataset import SkyDatasetDescription, SkyDataset


class SkyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, transforms):
        super(SkyDataModule, self).__init__()
        self.batch_size = batch_size
        self.transforms = transforms

        self.train_dataset = None
        self.test_dataset = None

        self.dims = None

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.description_path = os.path.join('skyhacks_hackathon_dataset', 'training_labels.csv')
        self.training_dataset_path = os.path.join('skyhacks_hackathon_dataset', 'training_images')

    def setup(self, stage: Optional[str] = None, train_test_split_ratio=0.85):
        train_dataset_description = SkyDatasetDescription(self.description_path)
        dataset = SkyDataset(self.training_dataset_path, train_dataset_description, self.transforms)
        dataset_length = len(dataset)
        train_dataset_length = int(dataset_length * train_test_split_ratio)
        train_test_split_size = [train_dataset_length, dataset_length - train_dataset_length]
        self.train_dataset, self.test_dataset = self.data_val = random_split(dataset, train_test_split_size)

        self.dims = dataset[0][0].shape

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        pass
