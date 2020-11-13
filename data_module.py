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
        self.val_dataset = None

        self.dims = None

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.description_path = os.path.join('skyhacks_hackathon_dataset', 'training_labels.csv')
        self.training_dataset_path = os.path.join('skyhacks_hackathon_dataset', 'training_images')

    def setup(self, stage: Optional[str] = None):
        train_dataset_description = SkyDatasetDescription(self.description_path)

        self.train_dataset = SkyDataset(self.training_dataset_path, train_dataset_description, self.transforms)

        self.dims = self.train_dataset[0][0].shape

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        pass