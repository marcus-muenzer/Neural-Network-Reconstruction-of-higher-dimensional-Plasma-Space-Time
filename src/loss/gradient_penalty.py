"""
NAME:
    gradient_penalty

DESCRIPTION:
    Computes the gradient penalty as addition for the discriminator loss in the Wasserstein GAN loss.
"""

import torch
import numpy
from torch.autograd import Variable
import globals.constants as const


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """
    Calculates the gradient penalty loss for WGAN GP.

    Parameters:
        discriminator: physics informed discriminator
        real_samples: tupel of real data
                      shape and content: (spacetimes, MHD states, eta(physical residuals))
                          eta is a function clamping each physical residual into the interval [0; 1]
                          where 0 relates to residuals of infinity and 1 to residuals of zero
                      dimensions of elements: tensor of spacetimes -> tensor.Size([batch size, const.dims_st])
                                              tensor of MHD states -> tensor.Size([batch size, const.dims_mhd_state])
                                              tensor of physical residuals -> tensor.Size([batch size, 1])

        fake_samples: tupel of fake/generated data
                      shape and content: (spacetimes, MHD states, eta(physical residuals))
                          eta is a function clamping each physical residual into the interval [0; 1]
                          where 0 relates to residuals of infinity and 1 to residuals of zero
                      dimensions of elements: tensor of spacetimes -> tensor.Size([batch size, const.dims_st])
                                              tensor of MHD states -> tensor.Size([batch size, const.dims_mhd_state])
                                              tensor of physical residuals -> tensor.Size([batch size, 1])

    Returns:
        gradient_penalty: scalar of the calculated gradient penalty
    """

    # Concatenates tuples as preparation for the interpolation
    interpol_input_real = torch.cat((real_samples[0], real_samples[1]), 1)
    interpol_input_fake = torch.cat((fake_samples[0], fake_samples[1]), 1)

    # Random weight term for interpolation between real and fake samples
    alpha = torch.tensor(numpy.random.random((interpol_input_real.size(0), const.dims_st + const.dims_mhd_state))).type(const.dtype)

    # Get random interpolation between real and fake samples
    interpolates = (alpha * interpol_input_real + ((1 - alpha) * interpol_input_fake)).requires_grad_(True)

    # Split concatenated tensor to match expected amount and dimensionality of the discriminator input
    st_interpolates, U_interpolates = torch.split(interpolates, [const.dims_st, const.dims_mhd_state], 1)
    # Use mean of residual tensor as residual interpolate
    # (1 + fake-residuals) / 2
    residual_interpolates = torch.div(real_samples[2].add(fake_samples[2]), 2)

    # Classify interpolated data
    d_interpolates = discriminator(st_interpolates, U_interpolates, residual_interpolates)

    fake = Variable(torch.ones(real_samples[0].shape[0], const.dims_mhd_state), requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
