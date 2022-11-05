"""
NAME
    kNN

DESCRIPTION
    This module provides a kNN class usable as a reconstructor.
    It just wrapps the KNeighborsRegressor.
"""

from sklearn.neighbors import KNeighborsRegressor


class kNNRegressor():
    def __init__(self, n_neighbors, leaf_size, weights):
        """
        Initializes the kNN.

        Parameters:
            n_neighbors: number of neighbors
            leaf_size: size of leafs
            weights: weighting criteria for predictions
                     options 'uniform', 'distance'

        Returns:
            None
        """

        self.model = KNeighborsRegressor(n_neighbors=n_neighbors,
                                         leaf_size=leaf_size,
                                         weights=weights)

    def train(self, st, mhd):
        """ Trains on the data """
        self.model.fit(st, mhd)

    def forward(self, st):
        """ Makes predictions """
        return self.model.predict(st)
