from builtins import object
from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass
import numpy as np
from scipy import signal as spsig
import scipy
import lmfit.models as lmod
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from . import fitfun
import csv

__author__ = "Peng Sun"
__license__ ="MIT License"

CI = 0.95 # confidence interval level=95%
Z = -scipy.stats.norm.ppf((1.0-CI)/2.0)

class MetaAnalyzer(with_metaclass(ABCMeta, object)):
    def __init__(self, device=None):
        self._device = device

    #
    # @abstractmethod
    # def cwt_peaks(self):
    #     pass
    #
    # @abstractmethod
    # def clustering_peaks(self):
    #     pass
    #
    # @abstractmethod
    # def partition_windows(self):
    #     pass
    #
    # @abstractmethod
    # def fit_resonances(self):
    #     pass

class RingAnalyzer(MetaAnalyzer):
    def __init__(self, FWHM_guess=0.1):
        super(RingAnalyzer, self).__init__(device='ring')

    @abstractmethod
    def readdata(self, inputfile=None):
        pass

    def deembed_envelop(self, Nwindow=10):
        self.ch1_filtered=np.convolve(self.ch1, np.ndarray.flatten(np.ones((1,Nwindow))/float(Nwindow)))[int(Nwindow/2):int(Nwindow/2)+len(self.lam)]
        self.ch1_filtered_dB=10.0*np.log10(np.abs(self.ch1_filtered))
        pwridx=self.ch1_filtered_dB>=max(self.ch1_filtered_dB)-7.0
        pwrwindow=self.ch1_filtered_dB[pwridx]
        lamwindow=self.lam[pwridx]
        pgc=np.polyfit(lamwindow,pwrwindow,2,full=True)
        self.gcplam=-pgc[0][1]/2.0/pgc[0][0]
        self.gcploss=-(pgc[0][2]-pgc[0][1]**2/4.0/pgc[0][0])/2.0
        self.gcbw1db=2.0*np.sqrt(np.abs(2.0/pgc[0][0]))
        self.gcfrnmean=pgc[1][0]/(lamwindow.max()-lamwindow.min())*np.diff(lamwindow).mean()
        self.gcfrnmax=max(abs(pwrwindow-np.polyval(pgc[0],lamwindow)))
        self.ch1_norm=self.ch1/self.ch1_filtered

        Nchop = int(Nwindow/2)
        self.lamchop=self.lam[Nchop:len(self.lam)-Nchop]
        self.ch1_norm_chop=self.ch1_norm[Nchop:len(self.lam)-Nchop]

        if self._hasDrop==True:
            self.ch2_norm=self.ch2/self.ch1_filtered
            self.ch2_norm_chop=self.ch2_norm[Nchop:len(self.lam)-Nchop]

    def cwt_peaks(self, width_range=[1,10], snr=5.0):
        width_guess = int(self.FWHM_guess/self.lamstep)
        wavelet_widths = np.logspace(*np.log10(width_range), num=100)
        self.cwt_guess=spsig.find_peaks_cwt(1-self.ch1_norm_chop, wavelet_widths, min_snr=snr)
        cwt_guess_duplist = []
        # remove duplicate in detected peaks: if spacing between two peaks <2*FWHM,
        # then the lower peak will be discarded
        for ii in range(len(self.cwt_guess)-1):
            if abs(self.cwt_guess[ii]-self.cwt_guess[ii+1])<=width_guess*2:
                if 1-self.ch1_norm_chop[self.cwt_guess[ii]] < 1-self.ch1_norm_chop[self.cwt_guess[ii+1]]:
                    cwt_guess_duplist.append(ii)
                else:
                    cwt_guess_duplist.append(ii+1)
        self.cwt_guess_nodup = np.delete(self.cwt_guess, cwt_guess_duplist)

        peaks=self.cwt_guess_nodup
        self.peakdifflist=np.zeros(len(peaks))
        for ii in range(len(peaks)):
            if ii==0: self.peakdifflist[ii] = abs(peaks[ii+1]-peaks[ii])
            else: self.peakdifflist[ii]=abs(peaks[ii]-peaks[ii-1])

        # np.savetxt('rawpeaks_withdiff.csv', np.transpose([self.lamchop[self.cwt_guess_nodup], self.peakdifflist, 1 - self.ch1_norm_chop[self.cwt_guess_nodup]]),
        #            header='lam,dist,trans', comments='', delimiter=',')

    def cluster_peaks(self,method='agglomerative'):
        # standardize data
        tr = 1-self.ch1_norm_chop[self.cwt_guess_nodup]
        trstd = ((tr-np.mean(tr))/np.std(tr))
        if np.std(self.peakdifflist)!=0:
            peakdiffstd = (self.peakdifflist-np.mean(self.peakdifflist))/np.std(self.peakdifflist)
        else:
            peakdiffstd = np.ones_like(self.peakdifflist)
        X= np.transpose([trstd,peakdiffstd])
        if method=='agglomerative':
            clustering = AgglomerativeClustering(n_clusters=2).fit(X)
        elif method=='spectral':
            clustering = SpectralClustering(n_clusters=2).fit(X)
        elif method=='kmeans':
            clustering = KMeans(n_clusters=2).fit(X)
        else:
            clustering = AgglomerativeClustering(n_clusters=2).fit(X)

        if np.mean(1-self.ch1_norm_chop[self.cwt_guess_nodup[np.argwhere(clustering.labels_==1)]]) > \
            np.mean(1-self.ch1_norm_chop[self.cwt_guess_nodup[np.argwhere(clustering.labels_==0)]]):
            signalidx,noiseidx = 1,0
        else: signalidx,noiseidx = 0,1

        self.cwt_guess_nodup_clean = self.cwt_guess_nodup[np.argwhere(clustering.labels_==signalidx)].flatten()

    def fit_all_resonances(self, Nwindow=50, FWHM_guess=0.1, EL=1125.0E3):
        resonance_all = []
        for ii in range(len(self.cwt_guess_nodup_clean)):
            if self.cwt_guess_nodup_clean[ii]-int(Nwindow/2)<0: lb=0
            else: lb=self.cwt_guess_nodup_clean[ii]-int(Nwindow/2)
            if self.cwt_guess_nodup_clean[ii]+int(Nwindow/2)>len(self.lamchop)-1: ub=len(self.lamchop)-1
            else: ub=self.cwt_guess_nodup_clean[ii]+int(Nwindow/2)
            x=self.lamchop[lb:ub]
            y=self.ch1_norm_chop[lb:ub]

            resonance_all.append(self.fit_resonance(self.lamchop[lb:ub], self.ch1_norm_chop[lb:ub],
                                                    port='thru', residx=ii, EL=EL))
            if self._hasDrop==True:
                resonance_all.append(self.fit_resonance(self.lamchop[lb:ub], 1-self.ch2_norm_chop[lb:ub],
                                                        port='drop', residx=ii, EL=EL))

        return resonance_all

    def fit_resonance(self, x, y, port='thru', residx=0, EL=1125.0E3):
        center_guess = x[np.argmin(y)]
        delta_lam_guess = self.FWHM_guess / np.sqrt(2.0)
        center_guess_band = delta_lam_guess * np.array([-1, 1]) + center_guess
        weights = np.ones(len(y))
        if port=='thru':
            mod = lmod.Model(fitfun.fitfun_thru)
            params_guess = {
                'r1': {'value': 0.9, 'min': 0.1, 'max': 1.0},
                'r2a': {'value': 0.9, 'min': 0.1, 'max': 1.0},
                'lam': {'value': center_guess, 'min': center_guess_band[0], 'max': center_guess_band[1]},
                'EL': {'value': EL, 'min': EL * 0.9, 'max': EL * 1.1}
            }
        elif port=='drop':
            mod = lmod.Model(fitfun.fitfun_drop)
            params_guess = {
                'A': {'value': 0.04, 'min': 1e-6, 'max': 0.5},
                'r1r2a': {'value': 0.9, 'min': 0.1, 'max': 1.0},
                'lam': {'value': center_guess, 'min': center_guess_band[0], 'max': center_guess_band[1]},
                'EL': {'value': EL, 'min': EL * 0.9, 'max': EL * 1.1}
            }

        mod.param_hints.update(params_guess)
        params = mod.make_params()
        fitout = mod.fit(y, x=x, params=params, weights=weights, method='nelder')
        ci_dict, ci_dict_idx = {}, 0
        for key in fitout.params.keys():
            ci_dict[key] = ci_dict_idx
            ci_dict_idx += 1
        fitdata = []
        fitdata.append(self.inputpath)
        fitdata.append(self.inputfile)
        fitdata.append(self.gcploss)
        fitdata.append(self.gcplam)
        fitdata.append(self.gcbw1db)
        for key in fitout.params.keys():
            fitout.params[key].stderr = np.sqrt(
                fitout.varcov[ci_dict[key], ci_dict[key]] * 2.0 * fitout.redchi) * 2.0 * Z
            fitdata.append(fitout.params[key].value)
            fitdata.append(fitout.params[key].stderr)
        fitdata.append(fitout.chisqr)
        fitdata.append(fitout.redchi)
        fitdata.append(fitout.aic)
        fitdata.append(fitout.bic)
        fitdata.append(port)
        fitdata.append(residx)

        return fitdata