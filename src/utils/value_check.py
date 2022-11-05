"""
NAME:
    value_check

DESCRIPTION:
    Checks parameter values from command line parser.
"""

import argparse


class CheckFraction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        """
        Initializes the CheckFraction class.

        Parameters:
            Multiple parameters that don't really matter because they get set automatically
            when the class is initialized by main.py.

            It never is called manually!

        Returns:
            None
        """

        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, fraction, option_string=None):
        """ Checks if fraction parameter is in range ]0; 1] """

        if not 0 < fraction <= 1:
            raise argparse.ArgumentError(self, "Value out of range. Parameter must meet condition: 0 < fraction <= 1")
        setattr(namespace, self.dest, fraction)


class CheckDelta(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        """
        Initializes the CheckDelta class.

        Parameters:
            Multiple parameters that don't really matter because they get set automatically
            when the class is initialized by main.py.

            It never is called manually!

        Returns:
            None
        """

        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, delta, option_string=None):
        """ Checks if delta parameter is > 0 """

        if not 0 < delta:
            raise argparse.ArgumentError(self, "Value out of range. Parameter must meet condition: 0 < delta")
        setattr(namespace, self.dest, delta)
