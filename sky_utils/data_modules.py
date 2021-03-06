# pytorch
from torch.utils.data import DataLoader, random_split

# pytorch lightning
import pytorch_lightning as pl

# torchvision
from torchvision.datasets import MNIST, CIFAR10
from pl_bolts.datamodules import CIFAR10DataModule
from torchvision import transforms

# numpy
import numpy as np

# standard
from typing import Union, List, Optional
import os

# utils
from sky_utils.datasets import SkyDatasetDescription, SkyDataset
from sky_utils.transform import train_preprocess


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir='data/mnist', batch_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()
        self.input_dims = None

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        # download data
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
        self.data_train, self.data_val = random_split(mnist_train, [55000, 5000])
        self.input_dims = self.data_train[0][0].shape
        self.data_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)


class Cifar10DataModule(pl.LightningDataModule):

    def __init__(self, config, data_dir='data/cifar10'):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.batch_size = self.config["hparams"]["batch_size"]
        self.transform = transforms.ToTensor()

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.input_dims = None

    def prepare_data(self):
        # download data
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        mnist_train = CIFAR10(self.data_dir, train=True, transform=self.transform)
        self.data_train, self.data_val = random_split(mnist_train, [45000, 5000])
        self.input_dims = self.data_train[0][0].shape
        self.config["hparams"]["input_dims"] = self.input_dims
        self.config["hparams"]["num_of_inputs"] = np.prod(self.input_dims)
        self.data_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)


class SkyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super(SkyDataModule, self).__init__()
        self.batch_size = batch_size

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.input_dims = None

        self.description_path = os.path.join('skyhacks_hackathon_dataset', 'training_labels.csv')
        self.training_dataset_path = os.path.join('skyhacks_hackathon_dataset', 'training_images')

    def setup(self, stage: Optional[str] = None, train_test_split_ratio=0.90):
        train_dataset_description = SkyDatasetDescription(self.description_path)
        dataset = SkyDataset(self.training_dataset_path, train_dataset_description, train_preprocess)
        dataset_length = len(dataset)
        train_dataset_length = int(dataset_length * train_test_split_ratio)
        train_test_split_size = [train_dataset_length, dataset_length - train_dataset_length]
        self.data_train, self.data_val = random_split(dataset, train_test_split_size)
        self.data_test = self.data_val
        self.input_dims = self.data_train[0][0].shape

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=False, num_workers=4)
