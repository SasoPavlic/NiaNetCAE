import math
import os
from typing import List, Optional, Union
import pandas as pd

from lightning.pytorch import LightningDataModule

from torch.utils.data import Dataset
from lightning.pytorch.utilities.data import DataLoader

from torchvision import transforms

from log import Log
from .nyu_transformer import *


class DatasetLoader(Dataset):
    def __init__(self, csv_file, transform=None, data_samples=100, batch_size=64):
        csv_file_path = os.getcwd() + csv_file
        Log.debug(csv_file_path)
        self.paths = pd.read_csv(csv_file_path, header=None,
                                 names=['image', 'depth'])

        if data_samples > 100:
            data_samples = 100

        all_data_samples, batched_data_samples = len(self.paths), len(self.paths)
        all_data_samples = int(all_data_samples * (data_samples / 100))


        if all_data_samples % batch_size == 0:
            batched_data_samples = all_data_samples
        else:
            batched_data_samples = all_data_samples - (all_data_samples % batch_size)

        if batched_data_samples <=0:
            batched_data_samples = batch_size
        self.paths = self.paths.head(int(batched_data_samples))
        self.transform = transform

        super(DatasetLoader, self).__init__()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        image = Image.open(self.paths['image'][index])
        depth = Image.open(self.paths['depth'][index])
        sample = {'image': image, 'depth': depth}

        if self.transform:
            sample = self.transform(sample)

        return sample


class NYUDataset(LightningDataModule):
    def __init__(
            self,
            data_samples: int = 100,
            data_path: str = '/data/',
            batch_size: int = 64,
            channel_dim: int = 3,
            horizontal_dim: int = 300,
            vertical_dim: int = 300,
            num_workers: int = 16,
            pin_memory: bool = False,
            train_size: int = 80,
            test_size: int = 10,
            val_size: int = 10,
            **kwargs,
    ):
        super().__init__()

        self.data_samples = data_samples
        self.data_path = data_path
        self.batch_size = batch_size
        self.channel_dim = channel_dim
        self.horizontal_dim = horizontal_dim
        self.vertical_dim = vertical_dim
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size

    def setup(self, stage: Optional[str] = None) -> None:
        __imagenet_pca = {
            'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
            'eigvec': torch.Tensor([
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ])
        }
        __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}

        train_transform = transforms.Compose(
            [
                Scale(240),
                RandomHorizontalFlip(),
                RandomRotate(5),
                CenterCrop([self.horizontal_dim, self.vertical_dim], [self.horizontal_dim, self.vertical_dim]),
                ToTensor(),
                Lighting(0.1, __imagenet_pca[
                    'eigval'], __imagenet_pca['eigvec']),
                ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                ),
                Normalize(__imagenet_stats['mean'],
                          __imagenet_stats['std'])

            ]
        )
        self.train_dataset = DatasetLoader(
            csv_file=self.data_path + "nyu2_train.csv",
            transform=train_transform,
            data_samples=self.data_samples,
            batch_size=self.batch_size)

        Log.debug(f"Train dataset size: {self.train_dataset.paths.shape[0]}")
        __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}

        test_transform = transforms.Compose(
            [
                Scale(240),
                CenterCrop([self.horizontal_dim, self.vertical_dim], [self.horizontal_dim, self.vertical_dim]),
                ToTensor(is_test=True),
                Normalize(__imagenet_stats['mean'],
                          __imagenet_stats['std'])
            ]

        )

        self.test_dataset = DatasetLoader(
            csv_file=self.data_path + "nyu2_test.csv",
            transform=test_transform,
            data_samples=self.data_samples,
            batch_size=self.batch_size)

        Log.debug(f"Test dataset size: {self.test_dataset.paths.shape[0]}")

    # TODO Implement re-usable datalaoder process
    # https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048

    def train_dataloader(self) -> DataLoader:
        data = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            # persistent_workers=True
        )

        return data

    def test_dataloader(self) -> DataLoader:
        data = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            # persistent_workers=True
        )

        return data
