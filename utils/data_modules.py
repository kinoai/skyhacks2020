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


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir='data/mnist', batch_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()
        self.dims = None

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
        self.dims = self.data_train[0][0].shape
        self.data_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)


class Cifar10DataModule(pl.LightningDataModule):

    def __init__(self, config, data_dir='data/cifar10', batch_size=256):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        # download data
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        mnist_train = CIFAR10(self.data_dir, train=True, transform=self.transform)
        self.data_train, self.data_val = random_split(mnist_train, [45000, 5000])
        self.config["hparams"]["input_dims"] = self.data_train[0][0].shape
        self.config["hparams"]["num_of_inputs"] = np.prod(self.data_train[0][0].shape)
        self.data_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)
