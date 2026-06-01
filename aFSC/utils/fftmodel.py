r"""
Author:         Lars de Jong
Date:           2024-05-14
Description:    The class FFTModel is used to calculate the FFT of a time series
                as well as to find the highest peaks in the FFT.

"""
import numpy as np

from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

class FFTModel():

    def __init__(self, threshold = 0.01, distance = None):
        self.threshold = threshold
        self.distance = distance


    def calculateFFT(self, 
                     data: np.ndarray, 
                     t: np.ndarray):
        r"""
        Method calculates the FFT of the time series

        Parameters
        ----------
        data : np.ndarray
            array with the data corresponding to the time points
        t : np.ndarray
            array with the time points

        Returns
        ----------
        List of:
        amplitudesOrig : array
            array with the original amplitudes
        amplitudes : array
            array with the norm of amplitudes
        freqs : array
            array with the frequencies
        """

        amplitudesOrig = 2*rfft(data)/data.shape[0]
        freqs = rfftfreq(data.shape[0], (t[-1]-t[0])/data.shape[0])*2*np.pi

        amplitudes = np.abs(amplitudesOrig)
        amplitudes[0] = amplitudes[0]/2

        return [amplitudesOrig, amplitudes, freqs]
    
    def performFFT(self, 
                   data: np.ndarray, 
                   t: np.ndarray, 
                   highFreq: float = 10):
        r"""
        Method calculate the FFT of the data as well as finds
        the highest peaks in the FFT

        Parameters
        ----------
        data : np.ndarray
            array with the data corresponding to the time points
        t : np.ndarray
            array with the time points
        highFreq : float optional
            highest frequency to be considered
        
        Returns
        ----------
        fftData : list
            list with arrays of the frequencies and amplitudes
        peakVal : list
            list with arrays of the frequencies and amplitudes of the peaks
            First is the frequency
            Seccond the peak amplitudes sorted by constant, cos and sin
            Third the peak amplitudes in absolute values
        """

        dataTrunc = data
        tTrunc = t

        fftData = self.getAmplitudeFrequency(dataTrunc, tTrunc, highFreq)

        peakInd, _ = find_peaks(fftData[1],
                                height=self.threshold * np.max(fftData[1]),
                                distance=self.distance)
        
        peakInd = np.append(0, peakInd)
        peakVal = [fftData[0][peakInd], fftData[2][peakInd], fftData[1][peakInd]]


        peakVal[1] = self.getAmplitude(peakVal[1])

        return fftData, peakVal

    def getAmplitudeFrequency(self, 
                              data: np.ndarray, 
                              t: np.ndarray, 
                              highFreq: float = 10):
        r"""
        Method that performs an FFT and returns the Amplitude Frequency response

        Parameters
        ----------
        data : np.ndarray
            array with the data corresponding to the time points
        t : np.ndarray
            array with the time points
        highFreq : float optional
            highest frequency to be considered

        Returns
        ----------
        freqs : array
            array with the frequencies
        amplitudes : array
            array with the amplitudes
        """
        
        amplitudesXOrig, amplitudes, freqs = self.calculateFFT(data, t)

        freqs = freqs[freqs < highFreq]
        amplitudes = amplitudes[0:len(freqs)]
        return [freqs, amplitudes, amplitudesXOrig] 
    
    def getAmplitude(self,complAmpl):
        r"""
        Method that returns the amplitude from the complex amplitude
        in sine and cosine form
        
        Parameters
        ----------
        complAmpl : np.ndarray
            array with the complex amplitude
            
        Returns
        ----------
        amplitude : np.ndarray
            array with the amplitudes
            first the constant, then cosine and sine
        """

        constAmp = np.real(complAmpl[0])/2
        cosAmp = np.real(complAmpl[1:])
        sinAmp = -np.imag(complAmpl[1:])

        amplitude = np.zeros(2*len(cosAmp)+1)
        amplitude[0] = constAmp
        amplitude[1::2] = cosAmp
        amplitude[2::2] = sinAmp

        return amplitude
    