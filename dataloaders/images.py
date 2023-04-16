import os
from typing import List, Optional, Union
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .transforms import *


class Transformer_train(Dataset):
    def __init__(self, csv_file, transform=None):
        """INTERNET CODE"""

        # TODO rename/replace to x_train
        csv_file_path = os.getcwd() + csv_file
        print(csv_file_path)
        self.paths = pd.read_csv(csv_file_path, header=None,
                                 names=['image', 'depth'])

        self.transform = transform

        """END OF INTERNET CODE"""

        # train_size = 0
        # if x_train.shape[0] % batch_size == 0:
        #     train_size = x_train.shape[0]
        # else:
        #     train_size = x_train.shape[0] - (x_train.shape[0] % batch_size)
        #
        # df_train = x_train.head(train_size)
        # self.y_train = y_train
        # self.y_train = torch.tensor(self.y_train[:].values)
        # self.x_train = torch.tensor(df_train[:].values)
        # https://stackoverflow.com/questions/50307707/convert-pandas-dataframe-to-pytorch-tensor
        super(Transformer_train, self).__init__()

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
            data_path: str,
            batch_size: int = 1,
            num_workers: int = 16,
            pin_memory: bool = False,
            train_size: int = 80,
            test_size: int = 10,
            val_size: int = 10,
            anomaly_label: bool = True,
            **kwargs,
    ):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.anomaly_label = anomaly_label

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    # TODO Implement re-usable datalaoder process
    # https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
    def train_dataloader(self) -> DataLoader:
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
                CenterCrop([304, 228], [152, 114]),
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
        self.train_dataset = Transformer_train(
            csv_file=self.data_path,
            transform=train_transform)

        data = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            # persistent_workers=True
        )

        return data
