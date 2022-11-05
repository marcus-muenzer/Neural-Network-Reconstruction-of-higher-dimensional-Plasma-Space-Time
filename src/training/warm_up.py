"""
NAME
    warm_up

DESCRIPTION
    Provides the possibility to pre-train models.
"""

import torch
import globals.constants as const


def warm_up(model, lr, n_epochs):
    """
    Does pure data warm-up training.
    Uses the Mean Squared Error as loss function.
    Does not track any metrics or parameters.

    Parameter:
        model: pytorch model
        lr: learning rate
        n_epochs: number of epochs

    Returns:
        model: warmed-up model
    """

    # Loss function
    loss = torch.nn.MSELoss()

    # Send loss to GPU if possible
    # If cuda GPU available the models would already be there
    if const.cuda:
        loss.cuda()

    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.5, 0.999))

    # ==============================
    # Warm up training
    # ==============================

    for epoch in range(n_epochs):
        for st_train, U_train in const.dataloader_train:

            optimizer.zero_grad()

            # Predict
            predictions = model(st_train)

            # Loss of predictions
            data_loss = loss(predictions, U_train)

            # Optimize model
            data_loss.backward()
            optimizer.step()

    return model
