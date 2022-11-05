"""
NAME
    MHDA_MLP

DESCRIPTION
    This module provides a magneto-hydrodynamics aided MLP classes
    usable as discriminators for the cGAN architecture.
"""

import torch.nn as nn
import torch
import globals.constants as const


class MHDA_MLP_Discriminator(nn.Module):
    def __init__(self, embedding_layers: [int], layers: [int], act_func='leakyReLu'):
        """
        Initializes the MHDA_MLP_Discriminator - a magneto-hydrodynamics aided MLP class.
        The MHDA_MLP embeds the the MHD states that should be classified using an (small) additional internal MLP.
        This embedding is passed to every layer of the MLP.
        This provides a external memory that constantly reminds the model of the input in a sophisticaded way.

        Parameters:
            embedding_layers[i]: size of ith embedding layer that encodes space-time information
            layers[i]: size of ith layer
            act_func: activation function. Options: 'leakyReLu', 'tanh'

        Returns:
            None
        """

        super(MHDA_MLP_Discriminator, self).__init__()

        # Activation function
        if act_func == 'leakyReLu':
            self.act_func = torch.nn.LeakyReLU()
        else:
            self.act_func = torch.tanh

        self.sigmoid = torch.nn.Sigmoid()

        # Create layers
        dims_mhd_aid = const.dims_st + const.dims_mhd_state

        # Embedding layers
        self.E_layers = torch.nn.ModuleList()
        for idx, layer_size in enumerate(embedding_layers):
            # First Layer
            if idx == 0:
                layer = torch.nn.Linear(dims_mhd_aid, layer_size)

            # Intermediate Layer(s)
            else:
                layer = torch.nn.Linear(embedding_layers[idx - 1], layer_size)

            self.E_layers.append(layer)

        # Prediction layers
        self.layers = torch.nn.ModuleList()
        for idx, layer_size in enumerate(layers):
            # First Layer
            if idx == 0:
                layer = torch.nn.Linear(dims_mhd_aid, layer_size)

            # Intermediate Layer(s)
            else:
                layer = torch.nn.Linear(embedding_layers[-1] + layers[idx - 1], layer_size)

            self.layers.append(layer)

        # Additional last Layer
        # Transforms to mhd dimensions
        layer = torch.nn.Linear(embedding_layers[-1] + layers[-1], const.dims_mhd_state)
        self.layers.append(layer)

    def forward(self, mhd):
        """ Makes predictions """

        # Embedding
        e = self.E_layers[0](mhd)
        for layer in self.E_layers[1:]:
            e = layer(e)
            e = self.act_func(e)

        # Prediction
        z = self.layers[0](mhd)
        for layer in self.layers[1:-1]:
            z = layer(torch.concat((e, z), 1))
            z = self.act_func(z)
        z = self.layers[-1](torch.concat((e, z), 1))
        z = self.sigmoid(z)
        return z


class Bilinear_MHDA_MLP_Discriminator(nn.Module):
    def __init__(self, layers: [int], act_func='leakyReLu'):
        """
        Initializes the Bilinear_MHDA_MLP_Discriminator - a magneto-hydrodynamics aided MLP class based on bilinear layers.
        The Bilinear_MHDA_MLP is based on the same principle as the MHDA_MLP.
        It constantly reminds the model of the original input by passing it to bilinear hidden layers
        and to the bilinear output layer.
        The training time is significantly higher than the training time of the plain MHDA_MLP.

        Parameters:
            layer[i]: size of ith layer
            act_func: activation function. Options: 'leakyReLu', 'tanh'

        Returns:
            None
        """

        super(Bilinear_MHDA_MLP_Discriminator, self).__init__()

        # Activation function
        if act_func == 'leakyReLu':
            self.act_func = torch.nn.LeakyReLU()
        else:
            self.act_func = torch.tanh

        self.sigmoid = torch.nn.Sigmoid()

        # Create layers
        dims_mhd_aid = const.dims_st + const.dims_mhd_state

        self.layers = torch.nn.ModuleList()
        for idx, layer_size in enumerate(layers):
            # First Layer
            if idx == 0:
                layer = torch.nn.Linear(dims_mhd_aid, layer_size)

            # Intermediate Layer(s)
            else:
                layer = torch.nn.Bilinear(dims_mhd_aid, layers[idx - 1], layer_size)

            self.layers.append(layer)

        # Last Layer
        layer = torch.nn.Bilinear(dims_mhd_aid, layers[-1], const.dims_mhd_state)
        self.layers.append(layer)

    def forward(self, mhd):
        """ Makes predictions """

        z = self.layers[0](mhd)
        z = self.act_func(z)
        for layer in self.layers[1:-1]:
            z = layer(mhd, z)
            z = self.act_func(z)
        z = self.layers[-1](mhd, z)
        z = self.sigmoid(z)
        return z
