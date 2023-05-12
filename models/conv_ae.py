import hashlib

import torch
import torch.distributions
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torchmetrics.image

from experiments.metrics import RMSELoss
from models.base import BaseAutoencoder
from models.types_ import *
from lightning.pytorch import LightningModule


class ConvAutoencoder(BaseAutoencoder, LightningModule):
    def __init__(self, solution, **kwargs):
        super(ConvAutoencoder, self).__init__()

        tensor_dim = kwargs['model_params']['tensor_dim']
        batch_size = kwargs['data_params']['batch_size']

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
        self.decoding_layers.append(nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.decoding_layers.append(nn.ReLU(inplace=True))

        # TODO Remove temporal assignments
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.num_epochs = kwargs['trainer_params']['max_epochs']
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

    def forward(self, input: Tensor, **kwargs) -> dict[str, list[Any] | Any]:
        """Flipping shape of tensors"""

        image = input['image']
        depth = input['depth']

        z = self.encode(input['image'])
        output = self.decode(z)

        return dict({'input': image, 'depth': depth, 'output': output})

    def loss_function(self, curr_device, **kwargs) -> dict:
        """
        Computes the AE loss function.
        :param kwargs:
        :return metrics:
        """
        criterionRMSE = RMSELoss()
        l1_criterion = nn.L1Loss()
        ssim = torchmetrics.image.StructuralSimilarityIndexMeasure().to(curr_device)

        input = kwargs['input']
        depth = kwargs['depth']
        output = kwargs['output']

        loss_depth = torch.abs(torch.log(torch.abs(output - depth) + 0.5).mean())
        loss_ssim = (1 - ssim(output, depth)) * 0.5

        loss_l1 = l1_criterion(output, depth)
        loss_RMSE = criterionRMSE(output, depth)

        loss = loss_depth + loss_ssim + loss_l1 + loss_RMSE

        metrics = dict(
            {'loss': loss,
             'loss_depth': loss_depth,
             'loss_ssim': loss_ssim,
             'loss_l1': loss_l1,
             'loss_RMSE': loss_RMSE})

        return metrics
