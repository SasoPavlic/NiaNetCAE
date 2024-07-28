import os
from typing import Optional

import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from log import Log
from .nyu_transformer import *


class BaseDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data).float()
        self.targets = torch.tensor(targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class BaseDataLoader(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            batch_size: int,
            num_workers: int,
            pin_memory: bool,
            train_size: float,
            val_size: float,
            test_size: float,
            data_percentage: float,
            **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.data_percentage = data_percentage

    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError("This method should be overridden by subclasses")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                          pin_memory=self.pin_memory)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          pin_memory=self.pin_memory)


class NYUDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = {
            'image': Image.open(self.data.iloc[index]['image']),
            'depth': Image.open(self.data.iloc[index]['depth']),
            'path': self.data.iloc[index]['image']
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class NYUDataLoader(BaseDataLoader):
    def __init__(
            self,
            data_path: str = 'data/',
            data_percentage: int = 100,
            batch_size: int = 32,
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
        super().__init__(
            data_path=data_path,
            data_percentage=data_percentage,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            **kwargs
        )
        self.horizontal_dim = horizontal_dim
        self.vertical_dim = vertical_dim
        self.channel_dim = channel_dim

    def setup(self, stage: Optional[str] = None) -> None:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        def load_csv_data(file_path):
            csv_file_path = os.path.join(base_path, file_path)
            Log.debug(f"CSV file path: {csv_file_path}")
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"File {csv_file_path} not found.")
            df = pd.read_csv(csv_file_path, header=None, names=['image', 'depth'])
            return df

        train_df = load_csv_data("data/nyu2_train.csv")
        test_df = load_csv_data("data/nyu2_test.csv")

        # Combine train and test dataframes
        combined_df = pd.concat([train_df, test_df])

        # Apply data percentage filter
        combined_df = combined_df.sample(frac=self.data_percentage / 100.0, random_state=42)

        combined_df = combined_df.sample(frac=1).reset_index(drop=True)  # Shuffle the combined dataset

        # Split the data into train, validation, and test sets
        train_data, test_data = train_test_split(combined_df, test_size=self.test_size)
        train_data, val_data = train_test_split(train_data, test_size=self.val_size)

        self.train_dataset = NYUDataset(train_data, transform=self.get_transform(train=True))
        self.val_dataset = NYUDataset(val_data, transform=self.get_transform(train=False))
        self.test_dataset = NYUDataset(test_data, transform=self.get_transform(train=False))

    def get_transform(self, train=True):
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

        if train:
            return transforms.Compose([
                Scale(240),
                RandomHorizontalFlip(),
                RandomRotate(5),
                CenterCrop([self.horizontal_dim, self.vertical_dim], [self.horizontal_dim, self.vertical_dim]),
                ToTensor(),
                Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
                ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                ),
                Normalize(__imagenet_stats['mean'], __imagenet_stats['std'])
            ])
        else:
            return transforms.Compose([
                Scale(240),
                CenterCrop([self.horizontal_dim, self.vertical_dim], [self.horizontal_dim, self.vertical_dim]),
                ToTensor(is_test=True),
                Normalize(__imagenet_stats['mean'], __imagenet_stats['std'])
            ])
