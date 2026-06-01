r"""
Author:         Lars de Jong
Date:           2025-03-05
Description:    Class handles the initial guess creation by 
                integration of the given object of the model 
                and performing a FFT to extract values.

"""
import logging

import numpy as np

from aFSC.utils.fftmodel import FFTModel

class initGuessModel:
    r"""
    Class handles the initial guess creation by integration of the given object of the model 
    and performing a FFT to extract values.
    """

    def __init__(self, 
                 modelObj: object,
                 hSetList: list,
                 initialVals: np.ndarray, 
                 tList: list, 
                 fftMinVal: float,
                 fftHighFreq: float,
                 fftDist: float,
                 saveFolderStr: str,
                 setValStr: str,
                 forced: bool,
                 logger: logging.Logger = None,
                 **kwargs):
        r"""
        Constructor for the initial guess model class.
        
        Parameters
        ----------
        modelObj : object
            The object of the model
        hSetList : list
            List with the sets of h values for each variable
        initialVals : np.ndarray
            Initial values for the integration
        tList : list
            List with start and end time of the integration
            and desired time steps
        fftMinVal : float
            Minimum value for the FFT, below which no peak is considered
        fftHighFreq : float
            Highest frequency to be considered in the FFT
        fftDist : float
            Distance in points between the peaks in the FFT
        saveFolderStr : str
            String where to save the time series data
        setValStr : str
            String with the set values for the model, used in the save string
        forced : bool
            Determines if the system is forced or not.
        logger : logging.Logger, optional
            The logger object, by default None

        kwargs : dict, optional
            Additional keyword arguments, such as 'timeScale' and 'perDataPoint'.
            - timeScale: float (default = 1.0) for scwitching between different units
            - perDataPoint: float (default = 1000) is the number of data points to consider for each period
        """
        self.modelObj = modelObj
        self.initialVals = initialVals
        self.tList = tList

        self.fftMinVal = fftMinVal
        self.fftHighFreq = fftHighFreq
        self.fftDist = fftDist

        self.hSetList = hSetList
        self.forced = forced

        initStr = ''.join(map(str, initialVals))
        tStr = ''.join(map(str, tList))
        self.saveStr = saveFolderStr + 'timeData_x0_' + initStr + '_t_' + tStr + '_' + setValStr + '.pkl'

        self.logger = logger

        self.timeScale = kwargs.get('timeScale', 1.0)
        self.perDataPoint = kwargs.get('perDataPoint', 1000)

        if logger is not None:
            logger.info("Initial guess model created.")

    def createInitialGuess(self):
        r"""
        Method creates the initial guess for the root finding algorithm.

        Returns
        ----------
        initialGuess : np.ndarray
            Initial guess for the root finding algorithm
        amp_s1 : float
            Amplitude of the first harmonic, used for the root finding algorithm
        """

        dataList, timeVec = self.getTimeData()

        _, peakData = self.getFFTData(dataList, 
                                      timeVec,
                                      self.fftMinVal,
                                      self.fftHighFreq,
                                      self.fftDist,
                                      self.logger)
        
        initialGuess, amp_s1 = self.getInitialGuess(peakData)

        return initialGuess, amp_s1

    def getTimeData(self):
        r"""
        Method handles the creation of the time integration data.

        Returns
        ----------
        intDataPer : np.ndarray
            Integration data with only complete oscillations
        intTimePer : np.ndarray
            Time data with only complete oscillations
        """
        timeData = self.modelObj.integrationModel(self.initialVals, self.tList)

        dataList = timeData[0]
        timeVec = timeData[1]

        intDataTrunc, intTimeTrunc = self.getPeriod(dataList, timeVec)

        if intDataTrunc is False:
            intDataPer = [dataList[i][-self.perDataPoint:] for i in range(len(dataList))]
            intTimePer = timeVec[-self.perDataPoint:]
        else:
            intDataPer = intDataTrunc
            intTimePer = intTimeTrunc
            for j in range(10):
                intDataPer = np.concatenate((intDataPer, intDataPer), axis=1)
                intTimePer = np.concatenate((intTimePer, intTimePer+intTimePer[-1]+intTimePer[1]-intTimePer[0]))
        
        return intDataPer, intTimePer/self.timeScale

    def getPeriod(self,
                  intData: np.ndarray, 
                  intTime: np.ndarray):
        r""" Methods extracts only complete oscillations
        from the integration data, which are later used for the FFT
        
        Parameters
        ----------
        intData : np.array
            Integration data
        intTime : np.array
            Time data
            
        Returns
        -------
        intDataPer : list
            Integration data with only complete oscillations
            with each variable data in a seperated list entry
        intTimePer : list
            Time data with only complete oscillations
        """

        persol = False
        targetVals = np.array([sublist[-1] for sublist in intData])
        for i in range(len(intData[0])-100, 0, -1):
            elc_cur = np.array([sublist[i-1] for sublist in intData])
            
            if all(np.abs(elc_cur-targetVals)/np.abs(targetVals) < 1e-3):
                intDataPer = np.array([sublist[i:] for sublist in intData])
                intTimePer = intTime[i:]-intTime[i]
                persol = True
                break
        
        if not persol:
            intDataPer = False
            intTimePer = False
        
        return intDataPer, intTimePer

    def getFFTData(self, 
                   intSol: list, 
                   intTime: np.ndarray, 
                   thresHold: float, 
                   highFreq: float, 
                   dist: float = None, 
                   logger: logging.Logger = None):
        r""" Method calculates the FFT data for the given solution

        Parameters
        ----------
        intSol : list
            list with the integration solutions for each variable
        intTime : np.ndarray
            list with the time points
        thresHold : float
            threshold for the FFT, when to consider a peak
        highFreq : float
            highest frequency to be considered
        dist : float, optional
            distance in points between the peaks
        logger : logging.Logger (default = None)
            instance of the logger

        Returns
        ----------
        fftData : list
            list with the FFT data, with the following structure:
            [frequencies, norm of amplitudes, amplitudes]
        peak : list
            list with the peak data, with the following structure:
            [peak frequencies, peak amplitudes, norm of peak amplitudes]
        """
        
        myFFT = FFTModel(thresHold, dist)
        fftData = []
        peak = []
        for valSol in intSol:
            fftDataVal, peakVal = \
                myFFT.performFFT(valSol, 
                                intTime, 
                                highFreq=highFreq)
            fftData.append(fftDataVal)
            peak.append(peakVal)

        if logger is not None:
            logger.info("FFT data generated.")

        return fftData, peak

    def getInitialGuess(self,
                        peakDataList: list):
        r""" 
        Method creates the initial guess for the root finding algorithm
        
        Parameters
        ----------
        peakDataList : list
            List with the peak data for each variable

        Returns
        ----------
        guessVec : np.ndarray
            Initial guess for the root finding algorithm
        amp_s1 : float
            Amplitude of the first harmonic, used for the root finding algorithm
        """
        
        for idxVar, hSet in enumerate(self.hSetList):

            peakData = peakDataList[idxVar]
            valAmp = peakData[1]
            valH = int((valAmp.shape[0]-1)/2)
            hDes = len(hSet)

            if idxVar == 0:
                if peakData[0].shape[0] == 1:
                    baseFreq = peakData[0][0]
                else:
                    baseFreq = peakData[0][1]
            
            if self.forced:
                if valH > hDes:
                    valAmpNew = valAmp[:hDes*2+1]
                else:
                    valAmpNew = np.zeros(hDes*2+1)
                    valAmpNew[:valAmp.shape[0]] = valAmp
                if idxVar == 0:
                    guessVec = valAmpNew
                    amp_s1 = valAmp[2]
                else:
                    guessVec = np.concatenate((guessVec, valAmpNew))

            else:
                if idxVar == 0:
                    if valH > hDes:
                        valAmpNew = np.concatenate((valAmp[:2], valAmp[3:hDes*2+1]))
                    else:
                        valAmpNew = np.zeros(hDes*2)
                        valAmpNew[:2] = valAmp[:2]
                        valAmpNew[2:valAmp.shape[0]-1] = valAmp[3:]
                    amp_s1 = valAmp[2]

                    guessVec = valAmpNew
                else:
                    if valH > hDes:
                        valAmpNew = valAmp[:hDes*2+1]
                    else:
                        valAmpNew = np.zeros(hDes*2+1)
                        valAmpNew[:valAmp.shape[0]] = valAmp
                    guessVec = np.concatenate((guessVec, valAmpNew))
        
        if not self.forced:
            guessVec = np.concatenate((guessVec, [baseFreq]))

        return guessVec, amp_s1