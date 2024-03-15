"""
**********
Exceptions
**********

Custom exceptions for NetComp.
"""

class UndefinedException(Exception):
    """Raised when matrix to be returned is undefined"""

class InputError(Exception):
    """Raised when input to algorithm is invalid"""

class KathleenError(Exception):
    """Raised when food is gross or it is too cold"""
