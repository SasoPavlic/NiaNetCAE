import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import math

"""
Convolutional output shape calculation is from this source:
https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/6
"""
def num2tuple(num):
    return num if isinstance(num, tuple) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    h_w, kernel_size, stride, pad, dilation = num2tuple(h_w), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(pad), num2tuple(dilation)
    pad = num2tuple(pad[0]), num2tuple(pad[1])

    h = math.floor((h_w[0] + sum(pad[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + sum(pad[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    return h, w


def convtransp2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1, out_pad=0):
    h_w, kernel_size, stride, pad, dilation, out_pad = num2tuple(h_w), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(pad), num2tuple(dilation), num2tuple(out_pad)
    pad = num2tuple(pad[0]), num2tuple(pad[1])

    h = (h_w[0] - 1) * stride[0] - sum(pad[0]) + dilation[0] * (kernel_size[0] - 1) + out_pad[0] + 1
    w = (h_w[1] - 1) * stride[1] - sum(pad[1]) + dilation[1] * (kernel_size[1] - 1) + out_pad[1] + 1

    return h, w


def conv2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1):
    h_w_in, h_w_out, kernel_size, stride, dilation = num2tuple(h_w_in), num2tuple(h_w_out), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(dilation)

    p_h = ((h_w_out[0] - 1) * stride[0] - h_w_in[0] + dilation[0] * (kernel_size[0] - 1) + 1)
    p_w = ((h_w_out[1] - 1) * stride[1] - h_w_in[1] + dilation[1] * (kernel_size[1] - 1) + 1)

    return (math.floor(p_h / 2), math.ceil(p_h / 2)), (math.floor(p_w / 2), math.ceil(p_w / 2))


def convtransp2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1, out_pad=0):
    h_w_in, h_w_out, kernel_size, stride, dilation, out_pad = num2tuple(h_w_in), num2tuple(h_w_out), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(dilation), num2tuple(out_pad)

    p_h = -(h_w_out[0] - 1 - out_pad[0] - dilation[0] * (kernel_size[0] - 1) - (h_w_in[0] - 1) * stride[0]) / 2
    p_w = -(h_w_out[1] - 1 - out_pad[1] - dilation[1] * (kernel_size[1] - 1) - (h_w_in[1] - 1) * stride[1]) / 2

    return (math.floor(p_h / 2), math.ceil(p_h / 2)), (math.floor(p_w / 2), math.ceil(p_w / 2))

def map_shape(gene):
    gene = np.array([gene])
    bins = np.array([0.0, 0.5])
    inds = np.digitize(gene, bins)

    if inds[0] - 1 == 0:
        return "SYMMETRICAL"

    elif inds[0] - 1 == 1:
        return "A-SYMMETRICAL"

    else:
        raise ValueError(f"Value not between boundaries 0.0 and 1.0. Value is: {inds[0] - 1}")

#TODO Remove if not used
def fibonacci_channels(n):
    golden_ratio = (1 + math.sqrt(5)) / 2  # Golden ratio

    # Generate Fibonacci numbers up to n
    fib_nums = [1, 1]
    for i in range(2, n):
        fib_nums.append(fib_nums[i-1] + fib_nums[i-2])

    # Scale the Fibonacci numbers by the golden ratio
    scaled_nums = [int(math.ceil(num * golden_ratio)) for num in fib_nums]

    return scaled_nums

#TODO Remove if not used
def halving_channels(input_size):
    results = []

    while input_size > 0:
        results.append(input_size)
        input_size = input_size // 2  # Floor division by 2

    return results

def map_layer_step(gene, channel_size, kernel_size, h_w, padding, stride):

    # h_w = conv2d_output_shape(h_w, kernel_size, stride, padding)
    # print(h_w)
    #
    # h_w = conv2d_output_shape(h_w, kernel_size, stride, padding)
    # print(h_w)
    #
    # h_w = conv2d_output_shape(h_w, kernel_size, stride, padding)
    # print(h_w)

    # channels = halving_channels(h_w[0])
    #
    # channels = [x for x in channels if x >= channel_size + channel_size]

    gene = np.array([gene])
    bins = []
    value = 1 / h_w[0]
    step = value
    for col in range(0, h_w[0]):
        bins.append(step)
        step += value
    bins[-1] = 1.01

    inds = np.digitize(gene, bins)


    return inds[0]

def map_num_layers(gene, layer_step, dimension):
    if layer_step == 0:
        max_layers = dimension
        return max_layers

    else:
        max_layers = round(dimension / layer_step)

    if max_layers == 1:
        return 1

    if max_layers == 2:
        return 2

    else:
        gene = np.array([gene])

        bins = []
        value = 1 / max_layers
        step = value
        for col in range(0, max_layers):
            bins.append(step)
            step += value
        bins[-1] = 1.01
        inds = np.digitize(gene, bins)

        return int(inds[0])

def map_activation(gene,architecture):
    gene = np.array([gene])
    bins = np.array([0.0, 0.125, 0.25, 0.375, 0.500, 0.625, 0.750, 0.875, 1.01])
    inds = np.digitize(gene, bins)

    if inds[0] - 1 == 0:
        architecture.activation_name = "ELU"
        return F.elu

    elif inds[0] - 1 == 1:
        architecture.activation_name = "RELU"
        return F.relu

    elif inds[0] - 1 == 2:
        architecture.activation_name = "Leaky RELU"
        return F.leaky_relu

    elif inds[0] - 1 == 3:
        architecture.activation_name = "RRELU"
        return F.rrelu

    elif inds[0] - 1 == 4:
        architecture.activation_name = "SELU"
        return F.selu

    elif inds[0] - 1 == 5:
        architecture.activation_name = "CELU"
        return F.celu

    elif inds[0] - 1 == 6:
        architecture.activation_name = "GELU"
        return F.gelu

    elif inds[0] - 1 == 7:
        architecture.activation_name = "TANH"
        return torch.tanh

    else:

        raise ValueError(f"Value not between boundaries 0.0 and 1.0. Value is: {inds[0] - 1}")

def map_num_epochs(gene):
    gene = np.array([gene])
    bins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.60, 0.7, 0.8, 0.9, 1.01])
    inds = np.digitize(gene, bins)

    return int(inds[0]) * 10 + 100

def map_learning_rate(gene):
    gene = np.array([gene])
    bins = []
    value = 1 / 1000
    step = value
    for col in range(0, 1000):
        bins.append(step)
        step += value
    bins[-1] = 1.01
    inds = np.digitize(gene, bins)
    lr = np.array(bins)[inds[0]]

    return round(lr, 2)


def calculate_convolution():
    input_tensor = torch.randn(64, 3, 304, 304).to('cuda')
    print(input_tensor.shape)
    # torch.Size([64, 3, 304, 304])

    layer = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1).to('cuda')
    output_tesnsor = layer(input_tensor)
    print(output_tesnsor.shape)
    # torch.Size([64, 16, 152, 152])


    layer = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1).to('cuda')
    output_tesnsor = layer(output_tesnsor)
    print(output_tesnsor.shape)
    # torch.Size([64, 32, 76, 76])


    layer = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1).to('cuda')
    test = layer(output_tesnsor)
    print(test.shape)
    # torch.Size([64, 32, 76, 76])


    layer = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1).to('cuda')
    output_tesnsor = layer(output_tesnsor)
    print(output_tesnsor.shape)
    # torch.Size([64, 64, 38, 38])

    layer = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1).to('cuda')
    output_tesnsor = layer(output_tesnsor)
    print(output_tesnsor.shape)
    # torch.Size([64, 64, 38, 38])

    layer = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1).to('cuda')
    output_tesnsor = layer(output_tesnsor)
    print(output_tesnsor.shape)
    # torch.Size([64, 256, 10, 10])

    layer = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1).to('cuda')
    output_tesnsor = layer(output_tesnsor)
    print(output_tesnsor.shape)
    # torch.Size([64, 512, 5, 5])

    layer = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1).to('cuda')
    output_tesnsor = layer(output_tesnsor)
    print(output_tesnsor.shape)
    # torch.Size([64, 1024, 3, 3])

    layer = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1).to('cuda')
    output_tesnsor = layer(output_tesnsor)
    print(output_tesnsor.shape)
    # torch.Size([64, 2048, 2, 2])

    layer = nn.Conv2d(2048, 4096, kernel_size=3, stride=2, padding=1).to('cuda')
    output_tesnsor = layer(output_tesnsor)
    print(output_tesnsor.shape)
    # torch.Size([64, 2048, 2, 2])

    layer = nn.Conv2d(4096, 8192, kernel_size=3, stride=2, padding=1).to('cuda')
    output_tesnsor = layer(output_tesnsor)
    print(output_tesnsor.shape)
    # torch.Size([64, 8192, 1, 1])

    layer = nn.Conv2d(8192, 16384, kernel_size=3, stride=2, padding=1).to('cuda')
    output_tesnsor = layer(output_tesnsor)
    print(output_tesnsor.shape)
    # torch.Size([64, 16384, 1, 1])


    pass



def generate_autoencoder (shape, channel_dim,h_w , layers, layer_step,architecture):

    input_shape = channel_dim
    output_shape = layer_step
    max_layers = layers

    if shape == "SYMMETRICAL":
        while layers != 0:
            architecture.encoding_layers.append(nn.Conv2d(in_channels=input_shape, out_channels=output_shape, kernel_size=3, stride=2, padding=1))
            h_w = conv2d_output_shape((input_shape, output_shape), 3, 2, 1)
            #print(f"Conv2D: {h_w}")
            architecture.encoding_layers.append(nn.ReLU(inplace=True))

            if layers == max_layers:
                architecture.decoding_layers.insert(0, nn.ConvTranspose2d(in_channels=output_shape,
                                                                          out_channels=1, kernel_size=3,
                                                                          stride=2, padding=1, output_padding=1))
            else:
                architecture.decoding_layers.insert(0, nn.ConvTranspose2d(in_channels=output_shape,
                                                                          out_channels=input_shape, kernel_size=3,
                                                                          stride=2, padding=1, output_padding=1))



            architecture.decoding_layers.insert(0,nn.LeakyReLU(negative_slope=0.2))
            h_w = convtransp2d_output_shape((output_shape, input_shape), 3, 2, 1)
            #print(f"Tran2D: {h_w}")
            architecture.bottleneck_size = output_shape

            layers = layers - 1
            input_shape = output_shape
            output_shape = output_shape + layer_step

        print("+++++++++++++++++++++++++++++++++++++++ARCHITECTURE+++++++++++++++++++++++++++++++++++++++")
        for layer in architecture.encoding_layers:
            if type(layer) == nn.Conv2d:
                print(f"Encoder: {conv2d_output_shape((layer.in_channels, layer.out_channels), layer.kernel_size, layer.stride, layer.padding)}")

        for layer in architecture.decoding_layers:
            if type(layer) == nn.ConvTranspose2d:
                print(f"Decoder: {convtransp2d_output_shape((layer.out_channels, layer.in_channels), layer.kernel_size, layer.stride, layer.padding)}")

        print(architecture)
        print("+++++++++++++++++++++++++++++++++++++++ARCHITECTURE+++++++++++++++++++++++++++++++++++++++")
        pass

    else:
        pass


def test(shape, layers, layer_step,architecture):
    if shape == "SYMMETRICAL":

        i = dataset_shape[1]
        z = dataset_shape[1] - layer_step

        while layers != 0:
            """Minimum depth reached"""
            if z < 1:
                architecture.encoding_layers.append(nn.Conv2d(in_channels=i, out_channels=z + 1, kernel_size=3, stride=2, padding=1))
                architecture.encoding_layers.append(nn.ReLU(inplace=True))
                architecture.decoding_layers.insert(0, nn.ConvTranspose2d(in_channels=z + 1, out_channels=i, kernel_size=3, stride=2, padding=1, output_padding=1))
                architecture.decoding_layers.append(nn.LeakyReLU(negative_slope=0.2))
                architecture.bottleneck_size = z + 1
                break

            architecture.encoding_layers.append(nn.Conv2d(in_channels=i, out_channels=z, kernel_size=3, stride=2, padding=1))
            architecture.encoding_layers.append(nn.ReLU(inplace=True))
            architecture.decoding_layers.insert(0, nn.ConvTranspose2d(in_channels=z, out_channels=i, kernel_size=3, stride=2, padding=1, output_padding=1))
            architecture.decoding_layers.append(nn.LeakyReLU(negative_slope=0.2))
            i = i - layer_step
            z = z - layer_step
            layers = layers - 1

        if len(architecture.encoding_layers) == 0:
            architecture.bottleneck_size = 0
        else:
            # TODO Introduce a betterway to get bottleneck size
            architecture.bottleneck_size = architecture.encoding_layers[-2].out_channels

    elif shape == "A-SYMMETRICAL":
        i = dataset_shape[1]
        z = dataset_shape[1] - layer_step

        if layers == 1 or layers == 2:
            architecture.encoding_layers.append(nn.Conv2d(in_channels=i, out_channels=z, kernel_size=3, stride=2, padding=1))
            architecture.encoding_layers.append(nn.ReLU(inplace=True))
            architecture.decoding_layers.insert(0, nn.ConvTranspose2d(in_channels=z, out_channels=i, kernel_size=3, stride=2, padding=1, output_padding=1))
            architecture.decoding_layers.append(nn.LeakyReLU(negative_slope=0.2))

        if layers >= 3:
            layers_encoder = random.randint(1, layers)
            layers_decoder = layers - layers_encoder

            encoder_counter = layers_encoder
            decoder_counter = layers_decoder

            if layers_decoder == 0:
                layers_encoder = layers_encoder - 1
                layers_decoder = 1

                encoder_counter = layers_encoder
                decoder_counter = layers_decoder

            while encoder_counter != 0:

                if z < 1:
                    architecture.encoding_layers.append(nn.Conv2d(in_channels=i, out_channels=z + 1, kernel_size=3, stride=2, padding=1))
                    architecture.encoding_layers.append(nn.ReLU(inplace=True))
                    architecture.bottleneck_size = z + 1
                    break

                architecture.encoding_layers.append(nn.Conv2d(in_channels=i, out_channels=z, kernel_size=3, stride=2, padding=1))
                architecture.encoding_layers.append(nn.ReLU(inplace=True))

                i = i - layer_step
                z = z - layer_step
                encoder_counter = encoder_counter - 1

            while decoder_counter != 0:

                if layers_decoder == 1:
                    architecture.decoding_layers.insert(0, nn.ConvTranspose2d(in_channels=i, out_channels=dataset_shape[1], kernel_size=3, stride=2, padding=1, output_padding=1))
                    architecture.decoding_layers.append(nn.LeakyReLU(negative_slope=0.2))
                    break

                layer_step = int((dataset_shape[1] - i) / decoder_counter)  # Make more complex logic
                last_i = i
                i = i + layer_step
                z = z + layer_step
                decoder_counter = decoder_counter - 1

                architecture.decoding_layers.append(nn.ConvTranspose2d(in_channels=last_i, out_channels=i, kernel_size=3, stride=2, padding=1, output_padding=1))
                architecture.decoding_layers.append(nn.LeakyReLU(negative_slope=0.2))

        if len(architecture.encoding_layers) == 0:
            architecture.bottleneck_size = 0
        else:
            #TODO Introduce a betterway to get bottleneck size
            architecture.bottleneck_size = architecture.encoding_layers[-2].out_channels

def map_optimizer(gene, architecture):
    gene = np.array([gene])
    bins = np.array([0.0, 0.167, 0.334, 0.50, 0.667, 0.834, 1.01])
    inds = np.digitize(gene, bins)

    """When AE does not have any layers"""
    if len(list(architecture.parameters())) == 0:
        architecture.optimizer_name = "Empty"
        return None

    if inds[0] - 1 == 0:
        architecture.optimizer_name = "Adam"
        return torch.optim.Adam(architecture.parameters(), lr=architecture.learning_rate)

    elif inds[0] - 1 == 1:
        architecture.optimizer_name = "Adagrad"
        return torch.optim.Adagrad(architecture.parameters(), lr=architecture.learning_rate)

    elif inds[0] - 1 == 2:
        architecture.optimizer_name = "SGD"
        return torch.optim.SGD(architecture.parameters(), lr=architecture.learning_rate)

    elif inds[0] - 1 == 3:
        architecture.optimizer_name = "RAdam"
        return torch.optim.RAdam(architecture.parameters(), lr=architecture.learning_rate)

    elif inds[0] - 1 == 4:
        architecture.optimizer_name = "ASGD"
        return torch.optim.ASGD(architecture.parameters(), lr=architecture.learning_rate)

    elif inds[0] - 1 == 5:
        architecture.optimizer_name = "RPROP"
        return torch.optim.Rprop(architecture.parameters(), lr=architecture.learning_rate)

    else:
        raise ValueError(f"Value not between boundaries 0.0 and 1.0. Value is: {inds[0] - 1}")
