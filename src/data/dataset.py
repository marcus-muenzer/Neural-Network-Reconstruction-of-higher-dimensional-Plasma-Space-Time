"""
NAME
    dataset

DESCRIPTION
    Provides a class that wraps (training) data.
    The class can be used for batch training using torch.utils.data.DataLoader.

"""

from torch.utils.data import Dataset
import globals.constants as const


class PlasmaDataset(Dataset):
    """
    This class wraps spacetime and MHD data and provides the functionality to extract batches out of the data.

    Parameters:
        Dataset: torch.utils.data.Dataset

    Returns:
        None
    """

    def __init__(self, st, U):
        """
        Initializes the PlasmaDataset class.

        Parameters:
            st: tensor of spacetimes
            U: tensor of MHD states

        Returns:
            None
        """

        device = 'cuda' if const.cuda else 'cpu'

        self.st = st.to(device)
        self.U = U.to(device)

    def __getitem__(self, index):
        """ Returns one item (spacetime and corresponding MHD state """
        return (self.st[index], self.U[index])

    def __len__(self):
        """ Returns the length of the dataset """
        return self.st.shape[0]
