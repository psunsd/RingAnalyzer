from RingAnalyzer import fitfun
import csv
from scipy.io import loadmat
import numpy as np
from RingAnalyzer.analyzer import RingAnalyzer
from math import pi
import os
from multiprocessing import Pool
from functools import partial

class PassiveRing(RingAnalyzer):
    def __init__(self, hasDrop=True, hasAdd=False, FWHM_guess=0.1, outputfile=''):
        self._hasDrop = hasDrop
        self._hasAdd = hasAdd
        self.FWHM_guess = FWHM_guess
        self._outputfile = outputfile

    def readdata(self, inputpath=None, inputfile=None):
        self.inputpath=inputpath
        self.inputfile=inputfile
        self.dat = loadmat(inputpath+inputfile)
        self.ch1 = np.reshape(self.dat['data'][0][0][0][0][0][0], (1, 6001))[0]  # thru
        self.ch2 = np.reshape(self.dat['data'][0][0][0][0][0][1], (1, 6001))[0]  # drop
        self.lam = np.reshape(self.dat['data'][0][0][1], (1, 6001))[0] * 1e9     # wavelength
        self.lamstep = np.mean(np.diff(self.lam))   # wavelength scan step

    def prepoutfile(self, outputfile=None):
        self.outputfile = outputfile
        with open(outputfile, 'w') as DF:
            spamwriterDF = csv.writer(DF, delimiter=',')
            spamwriterDF.writerow(['path', 'filename', 'r1', 'r1_ci', 'r2a', 'r2a_ci', 'lambda', 'lambda_ci', 'FWHM', 'FWHM_ci', 'chisqr','redchi','aic','bic','port','idx'])

def ring_worker(inputpath=None, inputfile=None, outputfile=None):
    # instance of PassiveRing analyzer class
    FWHM_guess=0.02 # guesstimated FWHM=20pm
    instance=PassiveRing(hasDrop=True, hasAdd=False, FWHM_guess=FWHM_guess, outputfile=outputfile)
    # read in the data
    instance.readdata(inputpath, inputfile)
    # FSR ~0.66nm from GC FP cavity; set smooth window to be 10x FSR
    Nwin=int(np.floor(int(0.66*10/instance.lamstep/2)/2))*2
    instance.deembed_envelop(Nwindow=Nwin)

    width_guess=int(FWHM_guess/instance.lamstep)
    width_range=[width_guess/4, width_guess*4] # wavelet width: FWHM/4 to FWHM*4
    instance.cwt_peaks(width_range=width_range, snr=6.0)

    # Electrical length of the ring=2*pi*neff*L in nanometer
    EL=2*pi*2.85*10.0E3
    instance.fit_all_resonances(Nwindow=width_guess*50, FWHM_guess=FWHM_guess, EL=EL)

def ring_processor(inputfile, inputpath, outputfile):
    # instance of PassiveRing analyzer class
    FWHM_guess=0.02 # guesstimated FWHM=20pm
    instance=PassiveRing(hasDrop=True, hasAdd=False, FWHM_guess=FWHM_guess, outputfile=outputfile)
    # read in the data
    instance.readdata(inputpath, inputfile)
    # FSR ~0.66nm from GC FP cavity; set smooth window to be 10x FSR
    Nwin=int(np.floor(int(0.66*10/instance.lamstep/2)/2))*2
    instance.deembed_envelop(Nwindow=Nwin)

    width_guess=int(FWHM_guess/instance.lamstep)
    width_range=[width_guess/4, width_guess*4] # wavelet width: FWHM/4 to FWHM*4
    instance.cwt_peaks(width_range=width_range, snr=6.0)

    # Electrical length of the ring=2*pi*neff*L in nanometer
    EL=2*pi*2.85*10.0E3
    device_all_resonance = instance.fit_all_resonances(Nwindow=width_guess*50, FWHM_guess=FWHM_guess, EL=EL)
    return device_all_resonance

def main():
    inputpath = '.\\sampleData\\'
    outputfile = 'ringdata.csv'
    inputfilelist = [file for file in os.listdir(inputpath) if file.endswith('.mat')]

    # prepare output data
    instance = PassiveRing()
    instance.prepoutfile(outputfile=outputfile)

    p = Pool(4)
    results = p.map(partial(ring_processor, inputpath=inputpath, outputfile=outputfile), inputfilelist)

    with open(outputfile,'a') as DF:
        spamwriterDF = csv.writer(DF, delimiter=',')
        for device in results:
            for res in device:
                spamwriterDF.writerow(res)

if __name__ == "__main__":
    main()