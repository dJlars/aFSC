r"""
Author:         Lars de Jong
Date:           2025-03-03
Description:    adaptive harmonic balance algorithm of the 
                electrophsyiological model of the beta cell. 

"""

import time
import logging
import argparse

import numpy as np

from aFSC.models.adaptiveHB import adaptiveHarmonicBalance
from betaCell.betaCellModel import betaCellfunctions
from aFSC.models.initGuessModel import initGuessModel
from betaCell.betaCellInt import integratedBetaCellSystem

class betaCellAHB(adaptiveHarmonicBalance):
    r"""
    This class implements the adaptive harmonic balance algorithm
    for the electrophysiological model of the beta cell.
    """

    def __init__(self,
                 indexSet: set,
                 config: argparse.Namespace,
                 amp_s1: float = 0,
                 saveFolderStr: str = "",
                 logger: logging.Logger = None,
                 **kwargs
                 ):
        r"""
        Constructor for the electrophysiological harmonic balance class.
        
        Parameters
        ----------
        indexSet : set
            The initial index set for the harmonics.
        config : argparse.Namespace
            The configuration parameters.
        amp_s1 :  float, optional
            The value the sine amplitude of the first harmonic 
            of the first variable is set to, by default Zero
        saveFolderStr : str, optional
            The folder where the results are saved, by default empty string
        logger : logging.Logger, optional
            The logger object, by default None
        
        **kwargs : dict
            Please see the documentation of the parent class
        """
        super().__init__(indexSet, 
                         config.nrEvalPts, 
                         forced=False,
                         logger=logger,
                         **kwargs)

        self.configFile = config
        self.betaCell = betaCellfunctions()

        self.amp_s1 = amp_s1
        self.atp = config.atp

        self.saveFolderStr = saveFolderStr

        if logger is not None:
            logger.info("Beta cell Harmonic Balance initialized.")


    def residualFunction(self, 
                         coeffs: np.ndarray) -> np.ndarray:
        r"""
        This function calculates the residual function of the 
        electrophysiological model of the beta cell.
        
        Parameters
        ----------
        coeffs : np.ndarray
            The coefficients of the harmonic balance expansion.
        
        Returns
        -------
        np.ndarray
            The residual function.
        """
        t0_total = time.time()
        
        varCoeffList, omega = self.splitCoeffVec(coeffs)

        posList, velList = self.getPosVel(varCoeffList, omega)
        
        timeResList = self.calculateElectricResiduum(posList, velList)
        
        residual = self.calcFreqResiduum(timeResList)

        t1_total = time.time()
        self.itrCt += 1
        print("Iteration: " + str(self.itrCt) + 
              " Function Eval Time: " + '{:.8f}'.format(t1_total-t0_total) +
              " Residuum Norm: " + '{:.6e}'.format(np.sqrt(np.sum(residual**2))), end='\r')
        return residual
    
    def calculateNonlinearExternalForce(self,
                                        posList: list,
                                        velList: list) -> np.ndarray:
        r"""
        This function calculates the nonlinear and external force.
        
        Parameters
        ----------
        posList : list
            List with the variables over one period.
        velList : list
            list with the derivatives of the variables over one period.
        
        Returns
        -------
        np.ndarray
            The nonlinear and external force.
        """
        v = posList[0]
        n = posList[1]
        ca = posList[2]

        ca_er = self.betaCell.Ca_er # µM
        atp = self.atp
        adp = self.betaCell.concentrationADP(atp) # µM

        i_ca, i_k, i_kca, i_katp = \
            self.betaCell.calculateIonCurrent(v, n, ca, adp, atp)

        n_inf = self.betaCell.activationFun(v, self.betaCell.n_in, \
                        self.betaCell.s_n)
        
        j_mem = self.betaCell.calcJmem(i_ca, ca)
        j_er = self.betaCell.calcJer(ca, ca_er)

        f_neV = - 1/self.betaCell.C_mem * (i_ca + i_k + i_kca + i_katp)
        f_nen = (n_inf-n)/self.betaCell.tau_n
        f_neCa = self.betaCell.f_Ca*(j_mem-j_er)
        
        return [f_neV, f_nen, f_neCa]

    def splitCoeffVec(self,
                      solCoeff: np.ndarray):
        r"""
        This method returns the coefficients and the base frequency.

        Parameters
        ----------
        solCoeff : np.ndarray
            vector with the solution coefficients

        Returns
        ---------
        hbCoeffList : list
            list with vectors for each variable with the 
            harmonic balance coefficients
        omega : float
            base frequency
        """

        varCoeffList = []
        endIdx = 0
        for i in range(self.nrVar):
            nrH = self.harmonicList[i]
            if i == 0:
                varCoeffVec = np.zeros(nrH*2+1)
                varCoeffVec[:2] = solCoeff[:2]
                varCoeffVec[2] = self.amp_s1
                varCoeffVec[3:] = solCoeff[2:2*nrH]
                endIdx += 2*nrH
            else:
                varCoeffVec = solCoeff[endIdx:endIdx+2*nrH+1]
                endIdx += 2*nrH+1
            varCoeffList.append(varCoeffVec)
                
        omega = solCoeff[-1]
        return varCoeffList, omega
    
    def assignUQParameters(self,
                           collocationPoint: np.ndarray):
        r"""
        This method assigns the collocation point to the harmonic balance object.

        Parameters
        ----------
        collocationPoint : np.ndarray
            The collocation point
        """
        self.atp = collocationPoint[0]

    def getInitialGuess(self, 
                        collocationPoint: list):
        r"""
        This methods returns a simple initial 
        guess for the coefficients.

        Parameters
        ----------
        collocationPoint : list
            The collocation Point

        Returns
        ----------
        np.ndarray
            initial guess vector
        """
        
        tList = [self.configFile.t0, 
                 self.configFile.t_end_tr* 60000, 
                 self.configFile.t_step_tr]

        initVals = [self.configFile.V_0, 
                    self.configFile.n_0, 
                    self.configFile.Ca_0]

        intModelObj = integratedBetaCellSystem(self.configFile)
        intModelObj.cellModel.atp = collocationPoint[0]
        myInitGuess = initGuessModel(intModelObj,
                                    self.harmonicSetList,
                                    initVals, 
                                    tList, 
                                    self.configFile.fftMinVal,
                                    self.configFile.fftHighFreq,
                                    self.configFile.fftDist,
                                    self.saveFolderStr,
                                    str(collocationPoint[0]),
                                    False,
                                    self.logger,
                                    timeScale=1000)
        initGuess, amp_s1 = myInitGuess.createInitialGuess()

        self.amp_s1 = amp_s1

        return [initGuess]
    
    def getCP(self):
        r"""
        This method returns the collocation point as a list.
        
        Returns
        ---------
        list
            List with the collocation point parameters.
        """
        return [self.atp]

    def getPosVel(self,
                  varCoeffList: list, 
                  omega: float):
        r"""
        This method calculates the lists of positions and velocities
        over one period.

        Parameters
        ----------
        varCoeffList : list
            list with the harmonic balance coefficients for each variable
        omega : float
            base frequency

        Returns
        ----------
        posList : list
            list with the positions over one period
        velList : list
            list with the velocities over one period
        """

        posList = []
        velList = []
        for idxVar, varCoeff in enumerate(varCoeffList):
            posVar, velVar, _ = self.calculatePositionVelocityAcceleration(varCoeff, 
                                                                           omega,
                                                                           idxVar)
            posList.append(posVar)
            velList.append(velVar)
        
        return posList, velList
    
    def calculateElectricResiduum(self,
                                  posList: list, 
                                  velList: list):
        r"""
        Method to calculate the residuum of the electric beta cell model.
        
        Parameters
        ----------
        pos : list
            The position of V, n and Ca of the electrical system.
        vel : list
            The velocity of V, n and Ca of the electrical system.
        
        Returns
        ----------
        residuum : np.ndarray
            The residuum of the electric beta cell model,
            with the residuum of V, n, and Ca stacked.
        """

        dvdt = velList[0]/1000 # V/ms
        dndt = velList[1]/1000 # 1/ms
        dcadt = velList[2]/1000 # µM/ms

        f_neList = self.calculateNonlinearExternalForce(posList,
                                                        velList)
        
        resTimeV = dvdt - f_neList[0]
        resTimen = dndt - f_neList[1]
        resTimeCa = dcadt - f_neList[2]
        
        return [resTimeV, resTimen, resTimeCa]

    def calcFreqResiduum(self,
                         timeResList: list):
        r"""
        This method calculates the frequency residuum.
        
        Parameters
        ----------
        timeResList : list
            list with the time residuals
            
        Returns
        ----------
        np.ndarray
            The frequency residuum
        """

        for idxRes, timeRes in enumerate(timeResList):
            E_hn_c = self.E_hn_c_List[idxRes]
            freqRes = E_hn_c @ timeRes
            if idxRes == 0:
                totalFreqRes = freqRes
            else:
                totalFreqRes = np.hstack([totalFreqRes, freqRes])

        return totalFreqRes
    
    def updateRandomParameters(self):
        r"""
        This methods updates the random parameters for 
        the random initial guess and returns them.
        """
        
        nrCoeffs = [(len(hSet)*2+1) for hSet in self.harmonicSetList]
        sAr1 = np.ones(nrCoeffs)*self.randomParametersVal[0]
        sAr2 = np.ones(nrCoeffs)*self.randomParametersVal[1]
        return np.stack([sAr1,sAr2], axis = 1)

