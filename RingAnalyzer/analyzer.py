import numpy as np
import matplotlib.pyplot as pl
from numpy import sin,cos,sqrt,exp,pi
from scipy.optimize import minimize
from scipy import signal as spsig
import scipy
import lmfit.models as lmod
from sklearn.cluster import AgglomerativeClustering
from builtins import object
from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass

__author__ = "Peng Sun"
__license__ ="MIT License"

class RingAnalyzer(with_metaclass(ABCMeta, object)):
    def __init__(self):
        self._hasThru = True
        self._hasDrop = False
        self._hasAdd = False

    @abstractmethod
    def deembed_envelop(self):
        pass

    @abstractmethod
    def cwt_peaks(self):
        pass

    @abstractmethod
    def clustering_peaks(self):
        pass

    @abstractmethod
    def partition_windows(self):
        pass

    @abstractmethod
    def fit_resonances(self):
        pass

