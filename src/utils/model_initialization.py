"""
NAME
    model_initialization

DESCRIPTION
    Initializes models.
"""

from models.reconstructors.MLP import MLP
from models.reconstructors.STB_MLP import STB_MLP, Bilinear_STB_MLP
from models.reconstructors.Transformer import Transformer
from models.discriminators.PID_MLP import PID_MLP
from models.discriminators.PID_Transformer import PID_Transformer
from models.discriminators.PID_MHDB_MLP import PID_MHDB_MLP, Bilinear_PID_MHDB_MLP
import globals.constants as const


def init_recon_model(model_type, embedding_layers, layers, transformer_neurons, act_func='tanh'):
    """
    Initializes a reconstruction model.

    Parameters:
        model_type: type of the model
                    options: 'MLP', 'Transformer', 'STB_MLP', 'Bilinear_STB_MLP'
        embedding_layers: list of embedding layers of the model
        layers: list of layer sizes of the model
        transformer_neurons: amount of neurons in one transformer layer
                             not used if model_type != 'Transformer'
        act_func: activation function
                  options: 'leakyReLu', 'tanh'

    Returns:
        model: reconstruction model
    """

    if model_type == 'MLP':
        model = MLP(layers, act_func)
    elif model_type == 'Transformer':
        model = Transformer(len(layers), transformer_neurons, act_func)
    elif model_type == 'STB_MLP':
        model = STB_MLP(embedding_layers, layers, act_func)
    elif model_type == 'Bilinear_STB_MLP':
        model = Bilinear_STB_MLP(layers, act_func)

    # Set dtype
    model.type(const.dtype)

    # Send to GPU if possible
    if const.cuda:
        model.cuda()

    return model


def init_disc_model(model_type, embedding_layers, layers, transformer_neurons, act_func='tanh', apply_sig=True):
    """
    Initializes a discrimination model for GANs.

    Parameters:
        model_type: type of the model
                    options: 'MLP', 'Transformer', 'MHDB_MLP', 'Bilinear_MHDB_MLP'
        embedding_layers: list of embedding layers of the model
        layers: list of layer sizes of the model
        transformer_neurons: amount of neurons in one transformer layer
                             not used if model_type != 'Transformer'
        act_func: activation function
                  options: 'leakyReLu', 'tanh'
        apply_sig: apply sigmoid function on the output

    Returns:
        model: discrimination model
    """

    if model_type == 'MLP':
        model = PID_MLP(layers, act_func, apply_sig)
    elif model_type == 'Transformer':
        model = PID_Transformer(len(layers), transformer_neurons, act_func, apply_sig)
    elif model_type == 'MHDB_MLP':
        model = PID_MHDB_MLP(embedding_layers, layers, act_func, apply_sig)
    elif model_type == 'Bilinear_MHDB_MLP':
        model = Bilinear_PID_MHDB_MLP(layers, act_func, apply_sig)

    # Set dtype
    model.type(const.dtype)

    # Send to GPU if possible
    if const.cuda:
        model.cuda()

    return model
