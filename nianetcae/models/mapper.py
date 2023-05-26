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


def conv2d_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1):
    h_w, kernel_size, stride, padding, dilation = num2tuple(h_w), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(padding), num2tuple(dilation)
    padding = num2tuple(padding[0]), num2tuple(padding[1])

    h = math.floor((h_w[0] + sum(padding[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + sum(padding[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    return h, w


def convtransp2d_output_shape(h_w, kernel_size=1, stride=1, padding=0, dilation=1, output_padding=0):
    h_w, kernel_size, stride, padding, dilation, output_padding = num2tuple(h_w), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(padding), num2tuple(dilation), num2tuple(output_padding)
    padding = num2tuple(padding[0]), num2tuple(padding[1])

    h = (h_w[0] - 1) * stride[0] - sum(padding[0]) + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
    w = (h_w[1] - 1) * stride[1] - sum(padding[1]) + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1

    return h, w


def conv2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1):
    h_w_in, h_w_out, kernel_size, stride, dilation = num2tuple(h_w_in), num2tuple(h_w_out), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(dilation)

    p_h = ((h_w_out[0] - 1) * stride[0] - h_w_in[0] + dilation[0] * (kernel_size[0] - 1) + 1)
    p_w = ((h_w_out[1] - 1) * stride[1] - h_w_in[1] + dilation[1] * (kernel_size[1] - 1) + 1)

    return (math.floor(p_h / 2), math.ceil(p_h / 2)), (math.floor(p_w / 2), math.ceil(p_w / 2))


def convtransp2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1, output_padding=0):
    h_w_in, h_w_out, kernel_size, stride, dilation, output_padding = num2tuple(h_w_in), num2tuple(h_w_out), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(dilation), num2tuple(output_padding)

    p_h = -(h_w_out[0] - 1 - output_padding[0] - dilation[0] * (kernel_size[0] - 1) - (h_w_in[0] - 1) * stride[0]) / 2
    p_w = -(h_w_out[1] - 1 - output_padding[1] - dilation[1] * (kernel_size[1] - 1) - (h_w_in[1] - 1) * stride[1]) / 2

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


# TODO Remove if not used
def fibonacci_channels(n):
    golden_ratio = (1 + math.sqrt(5)) / 2  # Golden ratio

    # Generate Fibonacci numbers up to n
    fib_nums = [1, 1]
    for i in range(2, n):
        fib_nums.append(fib_nums[i - 1] + fib_nums[i - 2])

    # Scale the Fibonacci numbers by the golden ratio
    scaled_nums = [int(math.ceil(num * golden_ratio)) for num in fib_nums]

    return scaled_nums


# TODO Remove if not used
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


def map_activation(gene, architecture):
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


def calculate_convolution(batch_size, channel_dim, h_w):
    input_tensor = torch.randn(batch_size, channel_dim, h_w[0], h_w[1]).to('cuda')
    print(input_tensor.shape)

    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 0
    dilation = 1

    layer = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation).to('cuda')
    output_tesnsor = input_tensor
    print(
        f" Funkcija: {conv2d_output_shape((output_tesnsor.shape[2], output_tesnsor.shape[3]), kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)}")
    output_tesnsor = layer(output_tesnsor)
    print(f"{type(layer)} {layer.in_channels, layer.out_channels}: {output_tesnsor.shape}")

    layer = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation).to('cuda')
    print(
        f" Funkcija: {conv2d_output_shape((output_tesnsor.shape[2], output_tesnsor.shape[3]), kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)}")
    output_tesnsor = layer(output_tesnsor)
    print(f"{type(layer)} {layer.in_channels, layer.out_channels}: {output_tesnsor.shape}")

    layer = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation).to('cuda')
    print(
        f" Funkcija: {conv2d_output_shape((output_tesnsor.shape[2], output_tesnsor.shape[3]), kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)}")
    output_tesnsor = layer(output_tesnsor)
    print(f"{type(layer)} {layer.in_channels, layer.out_channels}: {output_tesnsor.shape}")

    layer = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation).to('cuda')
    print(
        f" Funkcija: {conv2d_output_shape((output_tesnsor.shape[2], output_tesnsor.shape[3]), kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)}")
    output_tesnsor = layer(output_tesnsor)
    print(f"{type(layer)} {layer.in_channels, layer.out_channels}: {output_tesnsor.shape}")

    layer = nn.Conv2d(128, 256, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation).to('cuda')
    print(
        f" Funkcija: {conv2d_output_shape((output_tesnsor.shape[2], output_tesnsor.shape[3]), kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)}")
    output_tesnsor = layer(output_tesnsor)
    print(f"{type(layer)} {layer.in_channels, layer.out_channels}: {output_tesnsor.shape}")

    layer = nn.ConvTranspose2d(256, 128, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding, dilation=dilation).to('cuda')
    print(
        f" Funkcija: {convtransp2d_output_shape((output_tesnsor.shape[2], output_tesnsor.shape[3]), kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, output_padding=output_padding)}")
    output_tesnsor = layer(output_tesnsor)
    print(f"{type(layer)} {layer.in_channels, layer.out_channels}: {output_tesnsor.shape}")

    layer = nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding, dilation=dilation).to('cuda')
    print(
        f" Funkcija: {convtransp2d_output_shape((output_tesnsor.shape[2], output_tesnsor.shape[3]), kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, output_padding=output_padding)}")
    output_tesnsor = layer(output_tesnsor)
    print(f"{type(layer)} {layer.in_channels, layer.out_channels}: {output_tesnsor.shape}")

    layer = nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding, dilation=dilation).to('cuda')
    print(
        f" Funkcija: {convtransp2d_output_shape((output_tesnsor.shape[2], output_tesnsor.shape[3]), kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, output_padding=output_padding)}")
    output_tesnsor = layer(output_tesnsor)
    print(f"{type(layer)} {layer.in_channels, layer.out_channels}: {output_tesnsor.shape}")

    layer = nn.ConvTranspose2d(32, 16, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding, dilation=dilation).to('cuda')
    print(
        f" Funkcija: {convtransp2d_output_shape((output_tesnsor.shape[2], output_tesnsor.shape[3]), kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, output_padding=output_padding)}")
    output_tesnsor = layer(output_tesnsor)
    print(f"{type(layer)} {layer.in_channels, layer.out_channels}: {output_tesnsor.shape}")

    layer = nn.ConvTranspose2d(16, 1, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding, dilation=dilation).to('cuda')
    print(
        f" Funkcija: {convtransp2d_output_shape((output_tesnsor.shape[2], output_tesnsor.shape[3]), kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, output_padding=output_padding)}")
    output_tesnsor = layer(output_tesnsor)
    print(f"{type(layer)} {layer.in_channels, layer.out_channels}: {output_tesnsor.shape}")

    layer = nn.ConvTranspose2d(1, 1, kernel_size=kernel_size, stride=stride, padding=137, output_padding=output_padding,
                               dilation=dilation).to('cuda')
    output_tesnsor = layer(output_tesnsor)
    print(f"{type(layer)} {layer.in_channels, layer.out_channels}: {output_tesnsor.shape}")

    pass


# Fucntion which calculates the output shape of a convolutional layer
# It gets the tensor shape and tries to guess the correct output shape by modifiying kernel size, stride and padding
def calculate_last_layer(current, final):
    kernel_size = 3
    stride = 1
    padding = 0
    output_padding = 0
    dilation = 1

    print(f"Task for: {(current[0], current[1])} --> {(final[0], final[1])}")

    if current > final:
        while True:
            print(f"Trying padding: {padding}")
            test_shape = convtransp2d_output_shape(current,
                                                   kernel_size=kernel_size,
                                                   dilation=dilation,
                                                   stride=stride,
                                                   padding=padding,
                                                   output_padding=output_padding)

            print(f"Test shape: {test_shape}")
            if test_shape == final:
                return nn.ConvTranspose2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding,
                                          output_padding=output_padding, dilation=dilation).to('cuda')
            else:
                padding = padding + 1
                continue

    elif current < final:
        while True:
            print(f"Trying output padding: {output_padding}")
            test_shape = convtransp2d_output_shape(current,
                                                   kernel_size=kernel_size,
                                                   dilation=dilation,
                                                   stride=stride,
                                                   padding=padding,
                                                   output_padding=output_padding)

            print(f"Test shape: {test_shape}")
            if test_shape == final:
                return nn.ConvTranspose2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding,
                                          output_padding=output_padding, dilation=dilation).to('cuda')
            else:
                output_padding = output_padding + 1
                continue


def network_prunning(encoding_layers, decoding_layers, h_w):
    output_list = calculate_output_shapes(encoding_layers, decoding_layers, h_w)
    # Convolution can be stopped when the output is 1x1
    index = next((index for index, data in enumerate(output_list) if data == (1, 1)), None)

    if index is not None:
        del encoding_layers[index + 1:]
        del decoding_layers[:-index - 1:]


def calculate_output_shapes(encoding_layers, decoding_layers, h_w):
    tensor_shapes = list()
    output_tensors = list()
    input_temp = h_w[0]
    output_temp = h_w[1]

    for index, layer in enumerate(encoding_layers, start=0):
        if type(layer) == nn.Conv2d:
            conv_output = conv2d_output_shape((input_temp, output_temp), layer.kernel_size, layer.stride,
                                              layer.padding)
            input_temp = conv_output[0]
            output_temp = conv_output[1]
            tensor_shapes.append(conv_output)
            # print(f"Encoder {index}: {conv_output}")

    for index, layer in enumerate(decoding_layers, start=0):
        if type(layer) == nn.ConvTranspose2d:
            conv__trans_output = convtransp2d_output_shape(h_w=(input_temp, output_temp),
                                                           kernel_size=layer.kernel_size,
                                                           stride=layer.stride,
                                                           padding=layer.padding,
                                                           output_padding=layer.output_padding,
                                                           dilation=layer.dilation)
            input_temp = conv__trans_output[0]
            output_temp = conv__trans_output[1]
            tensor_shapes.append(conv__trans_output)
            # print(f"Decoder {index}: {conv__trans_output}")

    return tensor_shapes

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
