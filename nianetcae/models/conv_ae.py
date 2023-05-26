import hashlib
import time

import torch
import torch.distributions
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torchmetrics.image
from nianetcae.models.mapper import *

from nianetcae.experiments.metrics import RMSELoss
from nianetcae.models.base import BaseAutoencoder
from nianetcae.models.types_ import *
from lightning.pytorch import LightningModule


class ConvAutoencoder(BaseAutoencoder, nn.Module):
    def __init__(self, solution, **kwargs):
        super(ConvAutoencoder, self).__init__()
        """
        Dimensionality:
        y1: topology shape,
        y2: number of neurons per layer,
        y3: number of layers,
        y4: activation function
        y5: number of epochs,
        y6: learning rate
        y7: optimizer algorithm.
        """

        self.id = str(int(time.time())).strip()
        self.batch_size = kwargs['data_params']['batch_size']
        self.channel_dim = kwargs['data_params']['channel_dim']
        self.horizontal_dim = kwargs['data_params']['horizontal_dim']
        self.vertical_dim = kwargs['data_params']['vertical_dim']

        self.kernel_size = kwargs['model_params']['kernel_size']
        self.padding = kwargs['model_params']['padding']
        self.stride = kwargs['model_params']['stride']
        self.output_padding = kwargs['model_params']['output_padding']

        self.encoding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()

        self.topology_shape = map_shape(solution[0])
        self.topology_shape = "SYMMETRICAL"
        self.layer_step = map_layer_step(solution[1], self.channel_dim, self.kernel_size,
                                         (self.horizontal_dim, self.vertical_dim), self.padding, self.stride)
        self.num_layers = map_num_layers(solution[2], self.layer_step, kwargs['data_params']['horizontal_dim'])
        self.activation = map_activation(solution[3], self)
        self.num_epochs = map_num_epochs(solution[4])
        self.learning_rate = map_learning_rate(solution[5])

        self.bottleneck_size = 0

        """
        output_size = (input_size - kernel_size + 2 * padding) / stride + 1

        (304 - 3 + 2 * 1) / 2 + 1 = 152
        
        
        """

        self.generate_autoencoder()

        self.optimizer = map_optimizer(solution[6], self)
        self.get_hash()

        # Encoder
        # self.encoding_layers = nn.ModuleList()
        #
        # self.encoding_layers.append(nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1))
        # self.encoding_layers.append(nn.ReLU(inplace=True))
        # self.encoding_layers.append(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1))
        # self.encoding_layers.append(nn.ReLU(inplace=True))
        # self.encoding_layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
        # self.encoding_layers.append(nn.ReLU(inplace=True))
        #
        # # Decoder
        # self.decoding_layers = nn.ModuleList()
        #
        # self.decoding_layers.append(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1))
        # self.decoding_layers.append(nn.ReLU(inplace=True))
        # self.decoding_layers.append(nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1))
        # self.decoding_layers.append(nn.ReLU(inplace=True))
        # self.decoding_layers.append(nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1))
        # self.decoding_layers.append(nn.ReLU(inplace=True))

    def generate_autoencoder(self):
        # calculate_convolution(batch_size, channel_dim, h_w)

        input_shape = self.channel_dim
        output_shape = self.layer_step
        layers = self.num_layers
        max_layers = self.num_layers
        h_w = (self.horizontal_dim, self.vertical_dim)

        if self.num_layers != 0 and self.layer_step != 0:

            if self.topology_shape == "SYMMETRICAL":
                while layers != 0:
                    self.encoding_layers.append(
                        nn.Conv2d(in_channels=input_shape, out_channels=output_shape, kernel_size=3, stride=2,
                                  padding=1))

                    if layers == max_layers:
                        self.decoding_layers.insert(0, nn.ConvTranspose2d(in_channels=output_shape,
                                                                          out_channels=1, kernel_size=3,
                                                                          stride=2, padding=1, output_padding=1))
                    else:
                        self.decoding_layers.insert(0, nn.ConvTranspose2d(in_channels=output_shape,
                                                                          out_channels=input_shape, kernel_size=3,
                                                                          stride=2, padding=1, output_padding=1))

                    layers = layers - 1
                    input_shape = output_shape
                    output_shape = output_shape + self.layer_step

                print("+++++++++++++++++++++++++++++++++++++++START ARCHITECTURE "
                      "MODIFICATION+++++++++++++++++++++++++++++++++++++++")

                network_prunning(self.encoding_layers, self.decoding_layers, h_w)

                output_list = calculate_output_shapes(self.encoding_layers, self.decoding_layers, h_w, )

                last_layer = calculate_last_layer((output_list[-1][0], output_list[-1][0]), h_w)
                if last_layer is not None:
                    self.decoding_layers.append(last_layer)

                print(f"Layer outputs: {calculate_output_shapes(self.encoding_layers, self.decoding_layers, h_w, )}")
                self.num_layers = len(self.encoding_layers)
                self.bottleneck_size = self.encoding_layers[-1].out_channels

                print("+++++++++++++++++++++++++++++++++++++++END ARCHITECTURE "
                      "MODIFICATION+++++++++++++++++++++++++++++++++++++++")

            elif self.topology_shape == "A-SYMMETRICAL":
                # TODO Implement A-SYMMETRICAL
                pass

    def get_hash(self):

        self.hash_id = hashlib.sha1(str(str(self.topology_shape) +
                                        str(self.layer_step) +
                                        str(self.num_layers) +
                                        str(self.activation_name) +
                                        str(self.num_epochs) +
                                        str(self.learning_rate) +
                                        str(self.optimizer_name) +
                                        str(self.bottleneck_size) +
                                        str(self.encoding_layers) +
                                        str(self.decoding_layers)).encode('utf-8')).hexdigest()

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
            encoded = self.activation(result)
            # print(f"Encoder: {encoded.shape}")

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
            decoded = self.activation(result)
            # print(f"Decoder: {decoded.shape}")

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

    def loss_function(self, curr_device: str = 'cuda', **kwargs) -> dict:
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
