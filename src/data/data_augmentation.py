"""
NAME
    data_augmentation

DESCRIPTION
    Provides the functionality to augment MHD data for training.
"""

import torch
import joblib
import globals.constants as const


def augment_data(path_to_model):
    """
    Augments the training data to the size of the whole evaluation data.
    It changes the training data (const.st_train, const.U_train) in-place!

    Parameters:
        path_to_model: path to augmentation model
                       must be stored as .pth (pytorch model) or .joblib (no-pytorch model) file
                       no-pytorch models must have a method "forward" to map spacetime -> MHD state

    Returns:
        None
    """

    # Load model
    pytorch = False
    model = None
    if path_to_model.endswith('.pth'):
        model = torch.load(path_to_model)
        pytorch = True
    elif path_to_model.endswith('.joblib'):
        model = joblib.load(path_to_model)
    else:
        raise TypeError('Wrong file format: augmentation model must be either .pth or .joblib file')

    # Extend available spacetime for training
    const.st_train = const.st_eval

    # Augment data
    const.U_train = torch.zeros(0, const.dims_mhd_state)
    for st, _ in const.dataloader_eval:
        U_batch = None
        if pytorch:
            U_batch = model(st)
        else:
            U_batch = torch.from_numpy(model.forward(st.cpu())).cuda()
        const.U_train = torch.cat((const.U_train, U_batch))

    # Mark change in globals
    const.data_augmented = True
