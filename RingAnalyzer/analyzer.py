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

class RingAnalyzer(MetaAnalyzer):
    def __init__(self, FWHM_guess=0.1):
        super(RingAnalyzer, self).__init__(device='ring')

    @abstractmethod
    def readdata(self, inputfile=None):
        pass

    def analyze_gc(self, spectral_top=7.0):
        """ Analyze grating couplers
        The top portion of the transmission spectra in dB will be fitted to parabolic, from which the GC peak loss,
        peak wavelength, and the 1 dB bandwidth will be extracted.  Mean and max fit residual norms will be extracted,
        which represent the average and maximum ripples in the spectrum.

        Parameters
        spectral_top : float (optional)
            The top portion of the transmission spectra in dB that will be fitted to parabolic; default is 7.0

        Attributes
        gcplam : float
            Fitted peak wavelength in nm
        gcploss : float
            Fitted peak loss in dB
        gcbw1db : float
            Fitted 1 dB bandwidth in nm
        gcfrnmean : float
            Fitted residual norm, mean value in dB within the top portion of the spectrum
        gcfrnmax : float
            Fitted residual norm, max value in dB within the top portion of the spectrum
        """
        self.ch1_raw_dB=10.0*np.log10(np.abs(self.ch1))
        pwridx=self.ch1_raw_dB>=max(self.ch1_raw_dB)-spectral_top
        pwrwindow=self.ch1_raw_dB[pwridx]
        lamwindow=self.lam[pwridx]
        pgc=np.polyfit(lamwindow,pwrwindow,2,full=True)

        # import matplotlib.pyplot as pl; pl.plot(lamwindow, pwrwindow, 'b'); pl.plot(lamwindow, np.polyval(pgc[0], lamwindow), 'r'); pl.show()
        self.gcplam=-pgc[0][1]/2.0/pgc[0][0]
        self.gcploss=-(pgc[0][2]-pgc[0][1]**2/4.0/pgc[0][0])/2.0
        self.gcbw1db=2.0*np.sqrt(np.abs(2.0/pgc[0][0]))
        self.gcfrnmean=pgc[1][0]/(lamwindow.max()-lamwindow.min())*np.diff(lamwindow).mean()
        self.gcfrnmax=max(abs(pwrwindow-np.polyval(pgc[0],lamwindow)))

    def deembed_envelop(self, Nwindow=10):
        """ Deembed the envelop of grating couplers
        Transmission spectra of the through port will be smoothed by a moving-average window 
        of length Nwindow.  The smoothed transmission spectra will be used to deembed the through port and 
        drop port (if available) transmission spectra.

        Parameters
        Nwindow : int
            Moving average window length in the number of data point; in general it should be ~10x larger
            than the high frequency ripples in the raw spectra

        Attributes
        ch1_filtered : float
            Smoothed through port transmission in linear scale
        ch1_filtered_dB : float
            Smoothed through port transmission in dB
        ch1_norm : float
            Normalized through port transmission
        ch2_norm : float
            Normalized drop port transmission, if available
        lamchop : float
            Wavelength array after excluding the left and right paddings
        ch1_norm_chop : float
            Normalized through port transmission after excluding the left and right paddings
        ch2_norm_chop : float
            Normalized drop port transmission after excluding the left and right paddings
        """
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
        """ Detect peaks in normalized spectra using Continuous Wavelet Transform
        Normalized transmission spectra will go through CWT and then resonances are detected as ridge lines
        in the 2D time-frequency representation.

        Parameters
        width_range : length-2 list of float
            Lower and upper range of the wavelet width in the number of wavelength points used in the CWT.
            A rule-of-thumb is that wavelet width should range from FWHM/4 to FWHM*4 of the resonance
        snr : float
            Signal-to-noise ratio of the ridge line detection in CWT.  Note that the SNR depends on the noise
            in the spectra and ER of the resonances, so there is no universal value for all designs and tests

        Attributes
        cwt_guess : list of float
            CWT guessed indices of resonance peaks in the chopped transmission spectra
        cwt_guess_nodup : list of float
            CWT guessed resonance indices with duplicates removed: if spacing between two peaks is < 2*FWHM, then
            the peak with lower transmission will be discarded
        """

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

        self.cwt_guess_nodup_clean = self.cwt_guess_nodup

        # np.savetxt('rawpeaks_withdiff.csv', np.transpose([self.lamchop[self.cwt_guess_nodup], self.peakdifflist, 1 - self.ch1_norm_chop[self.cwt_guess_nodup]]),
        #            header='lam,dist,trans', comments='', delimiter=',')

    def cluster_peaks(self,method='agglomerative'):
        """ Distinguish true resonances from spurious resonances due to noise/distortion
        Use Clustering to distinguish true resonances from spurious resonances.  The clustering
        is on two dimensional space of normalized wavelength spacing and normalized transmission.

        Parameters
        method : str
            Method for clustering the points in the 2D space of normalized wavelength spacing and normalized 
            transmission.  Options are: 'agglomerative', 'spectral', 'kmeans'.  Default is 'agglomerative'.

        Attributes
        cwt_guess_nodup_clean : list of float
            True resonance indices with spurious resonances removed.
        """
        if not len(self.cwt_guess_nodup):
            print("CWT detected no peaks.")
        else:
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

    def fit_all_resonances(self, Nwindow=50, EL=1125.0E3):
        """ Fit all resonances in the current spectrum
        Use Clustering to distinguish true resonances from spurious resonances.  The clustering
        is on two dimensional space of normalized wavelength spacing and normalized transmission.

        Parameters
        Nwindow : int
            Window width of the partitioned transmission spectra for fitting.  A rule-of-thumb is that each window should 
            be at least 10x guessed FWHM of the resonances.
        EL : float
            Electrical length of the ring, which is 2*pi*ring_radius(in nm)*neff
            
        Attributes
        resonances_all : list of dictionaries
            True resonance indices with spurious resonances removed.
        """
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
        """ Fit an individual resonance 
        Use Clustering to distinguish true resonances from spurious resonances.  The clustering
        is on two dimensional space of normalized wavelength spacing and normalized transmission.

        Parameters
        x : array of float
            Wavelength of the partitioned window within which there is only one resonance
        y : array of float
            Transmission spectrum of the partitioned window within which there is only one resonance
        port : str
            Port of the spectrum.  Options are 'thru' and 'drop'.
        residx : int
            Index of the resonance in the spectrum
        EL : float
            Electrical length of the ring
            
        Attributes
        fitdata : list
            List of fitting results.
        """
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

        Tmin = min(y)
        ERraw = 10*np.log10(Tmin)
        HalfT = (1.0+Tmin)/2.0
        yleft = y[0:np.argmin(y)]
        yright = y[np.argmin(y):-1]
        xleft = x[0:np.argmin(y)]
        xright = x[np.argmin(y):-1]
        if (len(xleft)==0 or len(xright)==0):
            FWHMraw = 0.0
        else:
            FWHMraw = xright[np.argmin(np.abs(yright-HalfT))] - xleft[np.argmin(np.abs(yleft-HalfT))]

        fitdata.append(ERraw)
        fitdata.append(FWHMraw)

        return fitdata