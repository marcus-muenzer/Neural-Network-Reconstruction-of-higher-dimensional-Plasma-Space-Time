"""
NAME
    PID_MHDB_MLP

DESCRIPTION
    This module provides a magneto-hydrodynamics aided/biased MLP classes
    usable as physics-informed discriminators inside the cGAN architecture.
"""

import torch.nn as nn
import torch
import globals.constants as const


class PID_MHDB_MLP(nn.Module):
    def __init__(self, embedding_layers: [int], layers: [int], act_func='tanh', apply_sig=True):
        """
        Initializes the PID_MHDB_MLP - a magneto-hydrodynamics aided MLP class.
        It receives additional knowledge about the physical residuals in an extra input feature.
        The PID_MHDB_MLP embeds the the spacetimes, MHD states , and physical residuals using an (small) additional internal MLP.
        This embedding is passed as additional bias to every layer of the MLP.
        This provides a external memory that constantly reminds the model of the input in a sophisticaded way.

        Parameters:
            embedding_layers[i]: size of ith embedding layer that encodes space-time information
            layers[i]: size of ith layer
            act_func: activation function. Options: 'leakyReLu', 'tanh'
            apply_sig: apply sigmoid function on the output

        Returns:
            None
        """

        super(PID_MHDB_MLP, self).__init__()

        # Activation function
        if act_func == 'leakyReLu':
            self.act_func = torch.nn.LeakyReLU()
        else:
            self.act_func = torch.tanh

        self.beta = nn.Parameter(torch.randn(1))

        self.apply_sig = apply_sig
        self.sigmoid = torch.nn.Sigmoid()

        # Create layers
        dims_input = const.dims_st + const.dims_mhd_state + 1

        # Embedding layers
        self.E_layers = torch.nn.ModuleList()
        for idx, layer_size in enumerate(embedding_layers):
            # First Layer
            if idx == 0:
                layer = torch.nn.Linear(dims_input, layer_size)

            # Intermediate Layer(s)
            else:
                layer = torch.nn.Linear(embedding_layers[idx - 1], layer_size)

            self.E_layers.append(layer)

        # Prediction layers
        self.layers = torch.nn.ModuleList()
        for idx, layer_size in enumerate(layers):
            # First Layer
            if idx == 0:
                layer = torch.nn.Linear(dims_input, layer_size)

            # Intermediate Layer(s)
            else:
                layer = torch.nn.Linear(embedding_layers[-1] + layers[idx - 1], layer_size)

            self.layers.append(layer)

        # Additional last Layer
        # Transforms to mhd dimensions
        layer = torch.nn.Linear(embedding_layers[-1] + layers[-1], const.dims_mhd_state)
        self.layers.append(layer)

    def forward(self, st, U, residual):
        """ Makes predictions """

        # Concatenate input
        z = torch.cat((st, U, residual), 1)

        # Embedding
        bias = self.E_layers[0](z)
        for layer in self.E_layers[1:]:
            bias = layer(bias)
            bias = self.act_func(bias)

        # Prediction
        z = self.layers[0](z)
        for layer in self.layers[1:-1]:
            z = layer(torch.concat((bias, z), 1))
            z = self.act_func(self.beta * z)
        z = self.layers[-1](torch.concat((bias, z), 1))
        if self.apply_sig: z = self.sigmoid(z)

        return z


class Bilinear_PID_MHDB_MLP(nn.Module):
    def __init__(self, layers: [int], act_func='leakyReLu', apply_sig=True):
        """
        Initializes the Bilinear_PID_MHDB_MLP - a magneto-hydrodynamics aided/biased MLP class based on bilinear layers.
        The Bilinear_MHDB_MLP is based on the same principle as the PID_MHDB_MLP.
        It constantly reminds the model of the original input by passing it to bilinear hidden layers
        and to the bilinear output layer.
        The training time is significantly higher than the training time of the plain PID_MHDB_MLP.

        Parameters:
            layer[i]: size of ith layer
            act_func: activation function. Options: 'leakyReLu', 'tanh'
            apply_sig: apply sigmoid function on the output

        Returns:
            None
        """

        super(Bilinear_PID_MHDB_MLP, self).__init__()

        # Activation function
        if act_func == 'leakyReLu':
            self.act_func = torch.nn.LeakyReLU()
        else:
            self.act_func = torch.tanh

        self.beta = nn.Parameter(torch.randn(1))

        self.apply_sig = apply_sig
        self.sigmoid = torch.nn.Sigmoid()

        # Create layers
        dims_input = const.dims_st + const.dims_mhd_state + 1

        self.layers = torch.nn.ModuleList()
        for idx, layer_size in enumerate(layers):
            # First Layer
            if idx == 0:
                layer = torch.nn.Linear(dims_input, layer_size)

            # Intermediate Layer(s)
            else:
                layer = torch.nn.Bilinear(dims_input, layers[idx - 1], layer_size)

            self.layers.append(layer)

        # Last Layer
        layer = torch.nn.Bilinear(dims_input, layers[-1], const.dims_mhd_state)
        self.layers.append(layer)

    def forward(self, st, U, residual):
        """ Makes predictions """

        # Concatenate input
        bias = torch.cat((st, U, residual), 1)

        z = self.layers[0](bias)
        z = self.act_func(z)
        for layer in self.layers[1:-1]:
            z = layer(bias, z)
            z = self.act_func(self.beta * z)
        z = self.layers[-1](bias, z)
        if self.apply_sig: z = self.sigmoid(z)

        return z
