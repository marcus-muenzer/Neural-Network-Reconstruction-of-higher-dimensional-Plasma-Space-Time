"""
NAME
    STB_MLP

DESCRIPTION
    This module provides a spatio-temporal aided/biased MLP class
    usable as a reconstructor.
"""

import torch.nn as nn
import torch
import globals.constants as const


class STB_MLP(nn.Module):
    def __init__(self, embedding_layers: [int], layers: [int], act_func='tanh'):
        """
        Initializes the STB_MLP - a spatio-temporal aided/biased MLP class.
        The STB_MLP embeds the the spacetimes for which the MHD states should be reconstructed
        using an (small) additional internal MLP.
        This embedding is passed additional bias to every layer of the MLP.
        This provides a external memory that constantly reminds the model of the input in a sophisticaded way.

        Parameters:
            embedding_layers[i]: size of ith embedding layer that encodes space-time information
            layers[i]: size of ith layer
            act_func: activation function. Options: 'leakyReLu', 'tanh'

        Returns:
            None
        """

        super(STB_MLP, self).__init__()

        # Activation function
        if act_func == 'leakyReLu':
            self.act_func = torch.nn.LeakyReLU()
        else:
            self.act_func = torch.tanh

        self.beta = nn.Parameter(torch.randn(1))

        # Create layers
        # Embedding layers
        self.E_layers = torch.nn.ModuleList()
        for idx, layer_size in enumerate(embedding_layers):
            # First Layer
            if idx == 0:
                layer = torch.nn.Linear(const.dims_st, layer_size)

            # Intermediate Layer(s)
            else:
                layer = torch.nn.Linear(embedding_layers[idx - 1], layer_size)

            self.E_layers.append(layer)

        # Prediction layers
        self.layers = torch.nn.ModuleList()
        for idx, layer_size in enumerate(layers):
            # First Layer
            if idx == 0:
                layer = torch.nn.Linear(const.dims_st, layer_size)

            # Intermediate Layer(s)
            else:
                layer = torch.nn.Linear(embedding_layers[-1] + layers[idx - 1], layer_size)

            self.layers.append(layer)

        # Additional last Layer
        # Transforms to mhd dimensions
        layer = torch.nn.Linear(embedding_layers[-1] + layers[-1], const.dims_mhd_state)
        self.layers.append(layer)

    def forward(self, st):
        """ Makes predictions """

        # Embedding
        e = self.E_layers[0](st)
        for layer in self.E_layers[1:]:
            e = layer(e)
            e = self.act_func(e)

        # Prediction
        z = self.layers[0](st)
        for layer in self.layers[1:-1]:
            z = layer(torch.concat((e, z), 1))
            z = self.act_func(self.beta * z)
        z = self.layers[-1](torch.concat((e, z), 1))
        return z


class Bilinear_STB_MLP(nn.Module):
    def __init__(self, layers: [int], act_func='leakyReLu'):
        """
        Initializes the Bilinear_STB_MLP - a spatio-temporal aided MLP class based on bilinear layers.
        The Bilinear_STB_MLP is based on the same principle as the STB_MLP.
        It constantly reminds the model of the original input by passing it to bilinear hidden layers
        and to the bilinear output layer.
        The training time is significantly higher than the training time of the plain STB_MLP.

        Parameters:
            layer[i]: size of ith layer
            act_func: activation function. Options: 'leakyReLu', 'tanh'

        Returns:
            None
        """

        super(Bilinear_STB_MLP, self).__init__()

        # Activation function
        if act_func == 'leakyReLu':
            self.act_func = torch.nn.LeakyReLU()
        else:
            self.act_func = torch.tanh

        self.beta = nn.Parameter(torch.randn(1))

        # Create layers
        self.layers = torch.nn.ModuleList()
        for idx, layer_size in enumerate(layers):
            # First Layer
            if idx == 0:
                layer = torch.nn.Linear(const.dims_st, layer_size)

            # Intermediate Layer(s)
            elif idx != len(layers) - 1:
                layer = torch.nn.Bilinear(const.dims_st, layers[idx - 1], layer_size)

            # Last Layer
            else:
                layer = torch.nn.Bilinear(const.dims_st, layers[idx - 1], const.dims_mhd_state)

            self.layers.append(layer)

    def forward(self, st):
        """ Makes predictions """

        z = self.layers[0](st)
        z = self.act_func(z)
        for layer in self.layers[1:-1]:
            z = layer(st, z)
            z = self.act_func(self.beta * z)
        z = self.layers[-1](st, z)
        return z
