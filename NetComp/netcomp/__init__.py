"""
NetComp v0.2.2
==============

    NetComp is a Python package for comparing networks using pairwise distances,
    and for performing anomaly detection on a time series of networks. It is
    built on top of the NetworkX package. For details on usage and installation,
    see README.md.

"""

__author__ = 'Peter Wills <peter.e.wills@gmail.com>'
__version__ = '0.2.2'
__license__ = 'MIT'

import sys
if sys.version_info[0] < 3:
    m = "Python 3.x required (%d.%d detected)."
    raise ImportError(m % sys.version_info[:2])
del sys

from netcomp.linalg import *
import netcomp.linalg

from netcomp.distance import *
import netcomp.distance
