"""
NAME:
    loss_weighting

DESCRIPTION:
    Provides a helper function to weight loss terms.
"""


def get_lambda_from_share(share_1, share_2=None):
    """
    Calculates weights for loss terms for occupying defined shares.
    Works for two loss compositions:
    (1) Loss = loss_d + lambda_1 * loss_1
    (2) Loss = loss_d + lambda_1 * loss_1 + lambda_2 * loss_2

    Composition (1): consists of a total of two loss terms
    Composition (2): consists of a total of three loss terms

    Examples:
    Case (1) -> set share_2 = None:
        - share_1 = 0: loss_1 should get 0% of the overall shares
                              => lambda_1 = 0
        - share_1 = .5: loss_1 should get 50% of the overall shares (aka. equal weighting)
                              => lambda_1 = 1

    Case (2):
        - share_1 = 0, share_2 = 0: loss_1 and loss_2 should both get 0% of the overall shares
                                     => lambda_1 = lambda_2 = 0
        - share_1 = .33, share_2 = .33: loss_1 and loss_2 should each get 33% of the overall shares
                                     => lambda_1 = lambda_2 = 1

    Parameters:
        share_1: share of loss 1
        share_2: share of loss 2
                 None if loss composition is of type (1)

    Returns:
        lambda_1: weight for loss 1
        lambda_2: weight for loss 2
                  None if there is no loss 2
    """

    # Lambdas scheduled by curriculum method
    if not share_1: return None, None

    if not share_2:
        # Composition of two loss terms in total
        lambda_1 = share_1 / (1 - share_1)
        lambda_2 = None

    else:
        # Composition of three loss terms in total
        # lambda_1 = share_1 / (1 - share_1 - share_2) = lambda_2 / share_2 - lambda_2 - 1
        # lambda_2 = share_2 / (1 - share_1 - share_2) = lambda_1 / share_1 - lambda_1 - 1
        nom = share_1
        denom = 1 - share_1 - share_2
        lambda_1 = nom / denom
        lambda_2 = lambda_1 / share_1 - lambda_1 - 1

    return lambda_1, lambda_2
