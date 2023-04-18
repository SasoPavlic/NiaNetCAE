import hashlib
import random
import time

import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from tabulate import tabulate

from models.base import BaseAutoencoder
from models.types_ import *

import os
import urllib.request
from urllib.error import HTTPError

import lightning as L
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm.notebook import tqdm


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoding_layers = nn.ModuleList()

        self.encoding_layers.append(nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1))
        self.encoding_layers.append(nn.ReLU(inplace=True))
        self.encoding_layers.append(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1))
        self.encoding_layers.append(nn.ReLU(inplace=True))
        self.encoding_layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
        self.encoding_layers.append(nn.ReLU(inplace=True))

        # Decoder
        self.decoding_layers = nn.ModuleList()

        self.decoding_layers.append(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.decoding_layers.append(nn.ReLU(inplace=True))
        self.decoding_layers.append(nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.decoding_layers.append(nn.ReLU(inplace=True))
        self.decoding_layers.append(nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.decoding_layers.append(nn.ReLU(inplace=True))

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.num_epochs = 3
        self.hash_id = hashlib.sha1(str("ConvolutionalAutoencoder").encode('utf-8')).hexdigest()
        self.num_layers = 6
        self.bottleneck_size = 38

    def encode(self, x: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # encoded = x.view(x.size(0), -1)
        encoded = x

        for layer in self.encoding_layers:
            result = layer(encoded)
            # print(f"Encoder: {result.shape}")
            encoded = result  # self.activation(result)

        return encoded

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        decoded = z

        for layer in self.decoding_layers:
            result = layer(decoded)
            # print(f"Decoder: {result.shape}")
            decoded = result  # self.activation(result)

        """Flipping back to original shape"""
        reconstructed = decoded
        return reconstructed

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """Flipping shape of tensors"""

        z = self.encode(input)
        reconstructed = self.decode(z)

        return [reconstructed, input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the AE loss function.
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss

        details = {'loss': loss, 'Reconstruction_Loss': recons_loss.detach()}
        return details

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        reconstructed, input = self.forward(x)
        return [reconstructed, input]
