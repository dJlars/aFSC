r"""
Author:         Lars de Jong
Date:           2024-11-21
Description:    harmonic balance algorithm of the Duffing 

"""

import time
import logging
import argparse

import numpy as np

from aFSC.models.adaptiveHB import adaptiveHarmonicBalance


class duffingHB(adaptiveHarmonicBalance):
    r"""
    This class implements the adaptive harmonic balance algorithm 
    for the Duffing oscillator.
    """

    def __init__(self,
                 indexSet: set,
                 config: argparse.Namespace,
                 saveFolderStr: str = "",
                 logger: logging.Logger = None,
                 **kwargs
                 ):
        r"""
        Constructor for the Duffing harmonic balance class.
        
        Parameters
        ----------
        indexSet : set
            The initial index set for the harmonics.
        config : argparse.Namespace
            The configuration parameters.
        saveFolderStr : str, optional
            The folder where results are saved, by default ""
        logger : logging.Logger, optional
            The logger object, by default None
        
        **kwargs : dict
            Please see the documentation of the parent class
        """
        super().__init__(indexSet, 
                         config.nrEvalPts, 
                         forced=True,
                         logger=logger,
                         **kwargs)

        self.alpha = config.alpha
        self.beta = config.beta
        self.delta = config.delta
        self.gamma = config.gamma
        self.omega = config.omega

        self.configFile = config
        self.saveFolderStr = saveFolderStr

        if self.logger is not None:
            self.logger.info("Duffing Harmonic Balance initialized.")
        
    def residualFunction(self, 
                         coeffs: np.ndarray) -> np.ndarray:
        r"""
        This function calculates the residual function of the Duffing oscillator.
        
        Parameters
        ----------
        coeffs : np.ndarray
            The coefficients of the harmonic balance expansion.
        
        Returns
        -------
        residual : np.ndarray
            The residual function.
        """
        t0_total = time.time()
        x, v, a = self.calculatePositionVelocityAcceleration(coeffs,
                                              self.omega,
                                              0)
        
        tVec = np.linspace(0,2*np.pi/self.omega,x.shape[0])
        residual = self.E_hn_c_List[0] @ (a + self.delta* v + self.alpha* x + self.beta* np.sin(x) - self.gamma* np.cos(self.omega*tVec))

        tVec = np.linspace(0,2*np.pi/self.omega,x.shape[0])
        
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
        
        x = posList[0]

        f_n = self.beta*np.sin(x)

        t = np.linspace(0, 2*np.pi/self.omega, self.nrEvalPts)
        f_e = self.gamma* np.cos(self.omega*t)
        f_ne = f_n+f_e

        return [f_ne]
        
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
        return [solCoeff], self.omega
        
    def assignUQParameters(self,
                           collocationPoint: np.ndarray):
        r"""
        This method assigns the collocation point to the harmonic balance object.

        Parameters
        ----------
        collocationPoint : np.ndarray
            The collocation point
        """
        self.alpha = collocationPoint[0]
        self.beta = collocationPoint[1]
        self.delta = collocationPoint[2]
        self.gamma = collocationPoint[3]

    def getCP(self):
        r"""
        This method returns the collocation point as a list.
        
        Returns
        ---------
        list
            List with the collocation point parameters.
        """
        return [self.alpha, self.beta, 
                self.delta, self.gamma]

    def getInitialGuess(self, 
                        collocationPoint: list
                        ):
        r"""
        This methods returns a simple initial 
        guess for the coefficients.

        Parameters
        ----------
        collocationPoint : list
            Value for initial guess vector

        Returns
        ----------
        np.ndarray
            initial guess vector, with precalculated data 
            for each solution for an excitation frequency of 3 rad s^-1
        """
        nrCoeffs = sum([2*len(varHarmonics) +1 for varHarmonics in self.harmonicSetList])
        initGuess1 = np.zeros(nrCoeffs)
        initGuess1[1] = -2.359
        initGuess1[2] = 0.979

        initGuess2 = np.zeros(nrCoeffs)
        initGuess2[1] = 2.262
        initGuess2[2] = 0.885

        initGuess3 = np.zeros(nrCoeffs)
        initGuess3[1] = 0.1
        initGuess3[2] = 0.0015
        
        # return [initGuess1, 
        #         initGuess2, 
        #         initGuess3]
        return [initGuess1]
    
    def updateRandomParameters(self):
        r"""
        This methods updates the random parameters for 
        the random initial guess and returns them.
        """
        
        hSet = self.harmonicSetList[0]
        sAr1 = np.ones((len(hSet)*2+1))*self.randomParametersVal[0]
        sAr2 = np.ones((len(hSet)*2+1))*self.randomParametersVal[1]
        return np.stack([sAr1,sAr2], axis = 1)
