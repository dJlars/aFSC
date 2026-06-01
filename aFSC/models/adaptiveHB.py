r"""
Author:         Lars de Jong
Date:           2024-11-20
Description:    This class controls the adaptive harmonic balance algorithm

"""

import logging
import copy

from typing import Union

import numpy as np

from abc import ABC, abstractmethod
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from scipy.optimize import root, least_squares


class adaptiveHarmonicBalance(ABC):
    r"""
    This is the super class for all adaptive harmonic balance algorithms.
    """

    def __init__(self,
                 harmonicSetList: list,
                 nrEvalPts: int,
                 forced: bool,
                 logger: logging.Logger = None,
                 **kwargs
                 ):
        r"""
        Constructor for the adaptive harmonic balance class.
        
        Parameters
        ----------
        harmonicSetList : list
            A list of sets of harmonics for each harmonic
        nrEvalPts : int
            Number of evaluation points in on oscillation period.
        forced : bool
            Boolean to determine if the system is forced (True) or self-excited (False).
        logger : logging.Logger (default = None)
            Logger object
        
        kwargs : dict
            Additional keyword arguments, such as:
            - RootMethod: str (default = "hybr"), set the method for SciPys root
            - xTol: float (default = 1e-8), xTol for SciPys root
            - maxIter: int (default = 1000), maximum number of iterations for the adaptive harmonic balance algorithm
            - solThreshold: float (default = 0), threshold for root finding algorithm to reject trivial solution, 
            - getNewInitialGuess: bool (default = False), if True a new initial guess is generated via time integration
            - maxSamples: int (default = 1000), maximum number of samples for the root finding algorithm 
            - usedSamples: float (default = 0.1), fraction of samples to use for the root finding algorithm
            - AbsRangeRndGuess, NormalRndguess, RelRangeRndGuess: list, Method to calculate the Random guesses for root finding, based on keyword and structur of the provided list, 
              - AbsRangeRndGuess: Uniform random guess in range defined by min and max vectors provided by list
              - NormalRndguess: Normal random guess where mean and variance are defined by list
              - RelRangeRndGuess: Unniform random guess with boundaries set by percentages of the initial guess
            - maxH: int (default = 100), maximum number of harmonics to consider
            - hasJacobian: fun (default = None), function with defined Jacobian function
            - initVals: bool (default = None), boolean to recalculate initial guess for root
            - peakThreshold: list (default = 1e-1), threshold for the cut off value of the frequency spectrum determined by the excitation and nonlinear function
            - ampThreshold: float (default = 1e-1), threshold for the relative amplitude of the harmonics
            - onlyAdd: bool (default = None), add only harmonics but do not remove some
            - deflation: bool (default = False), enables the use of deflation technique
            - deflationFakeSolutions: list of two list, to avoid finding unwanted solutions these solutions are pprovided with set of harmonics and corresponding coefficients
            - maxDefSol: float (default = 1), maximum number of solutions to consider for deflation
            - deflationSolutionSelect: int (default = -1), index of the solution to select for deflation, -1 means the last solution
            - nonAdaptive: bool (default = False), if True the adaptive harmonic balance algorithm is not performed, but the solution is returned directly
            - sameH4all: bool (default = False), apply found harmonics to all variables
        """

        self._harmonicSetList = harmonicSetList
        self.maximalH = max([max(i) for i in self._harmonicSetList])
        if self.maximalH*4+1 > nrEvalPts:
            self._nrEvalPts = self.maximalH*4+1
        else:
            self._nrEvalPts = nrEvalPts
        self.calculateFourierTransformers()

        if not hasattr(self,'logger'):
            self.logger = logger

        self.forced = forced
        self.nrVar = len(harmonicSetList)
        self.itrCt = 0

        # root finding parameters
        self.methodList = kwargs.get("RootMethod",'hybr')
        self.xTolVal = kwargs.get("xTol",1e-8)
        self.maxIter = kwargs.get("maxIter", 1000)
        self.solThreshold = kwargs.get("solThreshold", 0)
        self.getNewInitialGuess = kwargs.get("getNewInitialGuess", False)
        
        # Randomize guess
        self.maxSamples = kwargs.get("maxSamples", 1000)
        self.usedSamples = kwargs.get("usedSamples", 0.1)
        methods = ["AbsRangeRndGuess","NormalRndguess","RelRangeRndGuess","RandomPush"]
        self.RandomGuess = None
        for method in methods:
            if method in kwargs:
                self.RandomGuess = method
                self.randomParametersVal = kwargs[method]
                self.randomParameters = self.updateRandomParameters()
        self.maxH = kwargs.get("maxH", 100)
        self.hasJacobian = kwargs.get("hasJacobian", None)

        self.initVals = kwargs.get("initVals", None)
        
        # adaptive parameters
        self.peakThreshold = kwargs.get("peakThreshold", 1e-1)
        self.ampThreshold = kwargs.get("ampThreshold", 1e-1)
        self.onlyAdd = kwargs.get("onlyAdd", False)
        
        # alternative deflation
        self.deflation = kwargs.get("deflation", False)
        self.deflationFakeSolutions = kwargs.get("deflationFakeSolutions", None)
        self.maxDefSol = kwargs.get("maxDefSol", 1)
        self.deflationSolutionList = []
        self.solCt = 0
        self.defl_p = 3
        self.defl_alpha = 1
        self.deflSolutionSelect = kwargs.get("deflSolutionSelect", -1)

        self.nonAdaptive = kwargs.get("nonAdaptive", False)
        self.sameH4all = kwargs.get("sameH4all", False)

        self.previousResults = []

        self.maxAmp = [None for i in range(self.nrVar)]
        self.firstAHB = [True for i in range(self.nrVar)]

        self.firstRefinement = None
    
        if self.logger is not None:
            self.logger.info("Adaptive Harmonic Balance initialized.")

    @property
    def harmonicSetList(self):
        return self._harmonicSetList
    
    @harmonicSetList.setter
    def harmonicSetList(self, harmonicSetList: int):
        self._harmonicSetList = harmonicSetList
        self.maximalH = max([max(i) for i in self._harmonicSetList])
        if self.maximalH*4+1 > self.nrEvalPts:
            self._nrEvalPts = int(self.maximalH*4+1)
        
        self.calculateFourierTransformers()
        if self.RandomGuess is not None:
            self.randomParameters = self.updateRandomParameters()

        if self.logger is not None:
            self.logger.info("The index set for the harmonics has been changed to " + str(harmonicSetList) + ".")
            self.logger.info("The Fourier transformers have been recalculated.")

    @property
    def nrEvalPts(self):
        return self._nrEvalPts

    @nrEvalPts.setter
    def nrEvalPts(self, nrEvalPts: int):
        if self.maximalH*4+1 > nrEvalPts:
            self._nrEvalPts = self.maximalH*4+1
        else:
            self._nrEvalPts = nrEvalPts
        self.calculateFourierTransformers()

        if self.logger is not None:
            self.logger.info("The number of evaluation points has been changed to " + str(nrEvalPts) + ".")
            self.logger.info("The Fourier transformers have been recalculated.")

    @abstractmethod
    def residualFunction(self):
        r"""
        This method calculates the residual function.
        """
        raise NotImplementedError
    
    @abstractmethod
    def assignUQParameters(self):
        r"""
        This method assigns the UQ parameters to the corresponding 
        parameters.
        """
        raise NotImplementedError
    
    @abstractmethod
    def getInitialGuess(self):
        r"""
        This method returns the initial guess for the root finding algorithm.
        """
        raise NotImplementedError

    @abstractmethod
    def getCP(self):
        raise NotImplementedError
    
    @abstractmethod
    def updateRandomParameters(self):
        raise NotImplementedError
    
    def solveRootProblem(self,
                         initialGuess: np.ndarray,
                         residualFun = None,
                         argsTuple = ()):
        r"""
        This method solves the root problem residualFunction 
        using the Newton-Raphson method.

        Parameters
        ----------
        initialGuess : np.ndarray
            The initial guess for the solution.
        residualFun : Callable function
            Function from which the roots are to be sought
        argsTuple : tuple
            tuple with arguments for the function
        
        Returns
        ----------
        sol.x : np.ndarray
            The solution of the root problem.
        """
        np.random.seed(100)

        if residualFun is None:
            residualFun = self.residualFunctionDeflation
        
        if not isinstance(self.methodList,list):
            self.methodList = [self.methodList]

        for method in self.methodList:
            self.methodStr = method
            if self.methodStr == 'hybr':
                optionsDict = {'maxfev':  500,
                                'xtol': self.xTolVal,
                                }
            elif self.methodStr == 'lm':
                optionsDict = {}
            else:
                optionsDict = {}
            
            sol = self.callrootfun(residualFun, initialGuess, argsTuple, optionsDict)
            
            if sol.success and np.max(sol.x) > self.solThreshold:
                if sol.message == "The solution converged.":
                    self.itrCt = 0
                    print("\n Solution found!") 
                    if self.logger is not None:
                        self.logger.info("The root problem has been solved.")

                    return sol.x
                else:
                    if any(np.abs(sol.x) > 1e-9):
                        print("\n " + sol.message)
                        print("Caution: Solution might be wrong")
                        self.itrCt = 0
                        if self.logger is not None:
                            self.logger.info(sol.message)
                            self.logger.info("Caution: Solution might be wrong")

                        return sol.x
        
        # use previous calcualted solutions as initial guess
        if len(self.previousResults)>0:
            print("\n Trying previous solutions!")
            previousSolInitGuessList = self.calcPreviousSolutionInitialGuess()

            preRes = np.array([np.linalg.norm(residualFun(prevInit)) \
                                for prevInit in previousSolInitGuessList])
            preResIdx = np.argsort(preRes)
            prevInitGuessListSort = [previousSolInitGuessList[i] for i in preResIdx]
            for prevInitialGuess in prevInitGuessListSort:
                sol = self.callrootfun(residualFun, prevInitialGuess, argsTuple, optionsDict)
                
                if sol.message == "The solution converged." and np.max(sol.x) > self.solThreshold:
                    self.itrCt = 0
                    print("\n Solution found!")
                    if self.logger is not None:
                        self.logger.info("The root problem has been solved.")

                    return sol.x
                self.itrCt = 0
                print("\nTesting next previous solution!")

        # calculate new initial guess via time integration
        if self.getNewInitialGuess:
            self.itrCt = 0
            print("\n Trying new initial guess!")
            cpPoint = self.getCP()
            newInitialGuess = self.getInitialGuess(cpPoint)
            sol = self.callrootfun(residualFun, newInitialGuess[-1], argsTuple, optionsDict)
            
            if sol.message == "The solution converged." and np.max(sol.x) > self.solThreshold:
                self.itrCt = 0
                print("\n Solution found!")
                if self.logger is not None:
                    self.logger.info("The root problem has been solved.")

                return sol.x

        if self.RandomGuess is not None:

            if self.RandomGuess == "RelRangeRndGuess":
                samplesOrg = np.array([np.random.uniform(self.randomParameters[idxG,0]*iniG,self.randomParameters[idxG,1]*iniG,self.maxSamples) for idxG, iniG in enumerate(initialGuess)])
            elif self.RandomGuess == "NormalRndguess":
                samplesOrg = np.array([np.random.normal(iniG,self.randomParameters[0],self.maxSamples) for iniG in initialGuess])
            elif self.RandomGuess == "AbsRangeRndGuess":
                samplesOrg = np.array([np.random.uniform(self.randomParameters[idxG,0],self.randomParameters[idxG,1],self.maxSamples) for idxG, iniG in enumerate(initialGuess)])
            
            # order for most promising sample
            residuals = np.zeros((samplesOrg.shape[1],))
            for idx in range(samplesOrg.shape[1]):
                self.itrCt = 0
                residuals[idx] = np.linalg.norm(residualFun(samplesOrg[:,idx],*argsTuple))
            
            # sort the residuals and samples
            sorted_indices = np.argsort(residuals)
            samplesSorted = samplesOrg[:, sorted_indices]
            if self.addZeroOnes:
                samples = np.hstack((np.zeros((samplesOrg.shape[0],1)),
                                np.ones((samplesOrg.shape[0],1)),
                                samplesSorted[:, :int(self.maxSamples*self.usedSamples)]))
            else:
                samples = samplesSorted[:, :int(self.maxSamples*self.usedSamples)]


            for idx in range(1,samples.shape[1]):
                self.itrCt = 0

                for method in self.methodList:
                    self.methodstr = method
                    if self.methodStr == 'hybr':
                        optionsDict = {'maxfev':  1000*initialGuess.shape[0],
                                        'xtol': self.xTolVal,
                                        'factor': 0.1}
                    elif self.methodStr == 'lm':
                        optionsDict = {}
                    else:
                        optionsDict = {}
                    
                    sol = self.callrootfun(residualFun, samples[:,idx], argsTuple, optionsDict)
                    
                    if sol.message == "The solution converged." and np.max(np.abs(sol.x)) > self.solThreshold:
                        self.itrCt = 0
                        print("\n Solution found!")
                        if self.logger is not None:
                            self.logger.info("The root problem has been solved with random input.")
                        return sol.x
                
        if self.logger is not None:
            self.logger.error("The root problem could not be solved.")
            self.logger.error(sol.message)

        raise ValueError("The root problem could not be solved.")
    
    def callrootfun(self, fun2Call, initGuess, argsTuple, optionsDict):
        r"""
        Internal function to select if the jacobian was provided for the problem
        """

        if self.hasJacobian is None:
            sol = root(fun2Call, 
                       initGuess,
                       argsTuple,
                       tol = self.xTolVal,
                       method = self.methodStr,
                       options = optionsDict)
        else:
            sol = root(fun2Call, 
                       initGuess,
                       argsTuple,
                       jac = self.hasJacobian,
                       tol = self.xTolVal,
                       method = self.methodStr,
                       options = optionsDict)

        return sol
    
    @abstractmethod
    def calculateNonlinearExternalForce(self):
        r"""
        This method calculates the nonlinear and external force.
        """
        raise NotImplementedError

    def calculateFourierTransformers(self):
        r"""
        This method calculates the Fourier transformers:
        - E_nh: inverse Fourier transform matrix
        - E_nh_c: Fourier transform matrix
        - derMat: derivative matrix
        """

        E_nh_List = []
        E_hn_c_List = []
        derMat_List = []
        harmonicList = []

        for indexSet in self._harmonicSetList:
            evalPts = np.arange(0,self._nrEvalPts)
            E_nh = np.zeros((self._nrEvalPts,len(indexSet)*2+1))
        
            E_nh[:,0] = 1
            
            cosMat = np.cos(2*np.pi/self._nrEvalPts*np.outer(evalPts,np.array(list(indexSet))))
            sinMat = np.sin(2*np.pi/self._nrEvalPts*np.outer(evalPts,np.array(list(indexSet))))

            E_nh[:,1::2] = cosMat
            E_nh[:,2::2] = sinMat

            E_hn_c = copy.deepcopy(E_nh.T)
            E_hn_c[0,:] = 0.5
            E_hn_c = E_hn_c*2/self._nrEvalPts

            derMat = np.zeros((len(indexSet)*2+1,len(indexSet)*2+1))
            j = 1
            for i in indexSet:
                derMat[j,j+1] = i
                derMat[j+1,j] = -i
                j += 2

            E_nh_List.append(E_nh)
            E_hn_c_List.append(E_hn_c)
            derMat_List.append(derMat)
            harmonicList.append(len(indexSet))

        self.E_nh_List = E_nh_List
        self.E_hn_c_List = E_hn_c_List
        self.derMat_List = derMat_List
        self.harmonicList = harmonicList

    def determineIndexSet(self, 
                          f_ne: np.ndarray,
                          omega: float,
                          solCoeff: np.ndarray,
                          ):
        r"""
        This method determines the index set by performing
        a Fourier transform on the nonlinear and external force.

        Parameters
        ----------
        f_ne : np.ndarray
            The nonlinear and external force.
        omega : float
            The base frequency of the solution, given in 1/s.
        solCoeff : np.ndarray
            The coefficients of the harmonic balance solution.

        Returns
        ----------
        indexSet : np.ndarray
            The index set with multiples of omega.
        coeffSet : np.ndarray
            Corresponding coefficients for the index set.
        """

        # extend the force to 4 times the length for better
        f_n1e = np.hstack((f_ne,f_ne,f_ne,f_ne))
        
        amplitudesOrig = 2*rfft(f_n1e)/f_n1e.shape[0]
        freqs = rfftfreq(f_n1e.shape[0], 4*2*np.pi/omega/f_n1e.shape[0])*2*np.pi

        amplitudes = np.abs(amplitudesOrig)
        amplitudes[0] = amplitudes[0]/2

        # define distance, so only multiples of the base frequency are determined
        if self.firstRefinement is None:
            self.firstRefinement = np.argmin(np.abs(freqs-omega))

        # extract relevant harmonics and corresponding amplitudes
        peakInd, _ = find_peaks(amplitudes,
                            height=self.peakThreshold*np.max(amplitudes),
                            distance = self.firstRefinement
                            )
    
        addPeakInd, _ = find_peaks(amplitudes,
                                   height=self.peakThreshold*np.max(amplitudes)*0.01,
                                   distance=self.firstRefinement
                                   )  
        
        # createindex set and new coefficients
        if peakInd.size == 0:
            indexSet = {}
            coeffSet = solCoeff
        else:
            peakIndSet = set(peakInd)
            addPeakIndSet = set(addPeakInd)

            unionPeakIndSet = peakIndSet.union(addPeakIndSet)
            unionPeakInd = np.array(sorted(list(unionPeakIndSet)))

            freqSet = freqs[unionPeakInd]
            ampSet = amplitudesOrig[unionPeakInd]
                
            if len(addPeakInd) > len(peakInd):
                if addPeakInd[0] < peakInd[0]:
                    peakInd = addPeakInd
                else:
                    peakInd = np.concatenate((peakInd,np.array([addPeakInd[len(peakInd)]])))
            else:
                if len(peakInd) > 1:
                    addIndex = peakInd[-1] + (peakInd[-1] - peakInd[-2])
                    peakInd = np.concatenate((peakInd,np.array([addIndex]))) 
            
            if freqSet[0] == 0:
                cosCoeff = np.real(ampSet[1:])
                sinCoeff = -np.imag(ampSet[1:])
            else:
                cosCoeff = np.real(ampSet)
                sinCoeff = -np.imag(ampSet)
            coeffSet = np.zeros(2*len(cosCoeff))
            coeffSet[::2] = cosCoeff
            coeffSet[1::2] = sinCoeff

            # determine the index set
            indexSet = np.rint(freqSet/omega)

        return set(indexSet), coeffSet
    
    def calculatePositionVelocityAcceleration(self,
                                              coeffsOrg: np.ndarray,
                                              omega: float,
                                              index: int
                                              ):
        r"""
        This method calculates the position, velocity and acceleration
        of the system using the Fourier transformers.

        Parameters
        ----------
        coeffs : np.ndarray
            The coefficients of the harmonic balance.
        omega : float
            The base frequency of the solution.
        index : int
            index for the corresponding variable

        Returns
        ----------
        x : np.ndarray
            The position of the system.
        xdot : np.ndarray
            The velocity of the system.
        xddot : np.ndarray
            The acceleration of the system.
        """
        coeffs = copy.deepcopy(coeffsOrg)
        coeffs[np.abs(coeffs)<np.finfo(float).eps] = 0
        x = self.E_nh_List[index] @ coeffs
        xdot = omega* self.E_nh_List[index] @ \
            (self.derMat_List[index] @ coeffs)
        xddot = omega**2 * self.E_nh_List[index] @ \
            (self.derMat_List[index] @ (self.derMat_List[index] @ coeffs))
    
        return x, xdot, xddot
    
    @abstractmethod
    def splitCoeffVec(self,
                      solCoeff: np.ndarray):
        r"""
        This methods splits either the solution of just simply 
        returns the coefficients and the base frequency.
        """
        raise NotImplementedError

    def adaptiveHB(self,
                   initialGuess: Union[list, np.ndarray]):
        r"""
        This method performs the adaptive harmonic balance algorithm.

        Parameters
        ----------
        initialGuess : np.ndarray or list
            The initial guess for the solution.

        Returns
        ----------
        solCoeff : np.ndarray
            The solution of the adaptive harmonic balance algorithm.
        self.harmonicSetList : list
            List with the current harmonic sets list
        hbAmpRatioListList : list
            List with the amplitude ratio for each variable at each iteration step
        """

        self.firstAHB = [True for i in range(self.nrVar)]

        # perform root finding in deflation mode
        self.calculateDeflationsSolutions(initialGuess)
        
        if self.logger is not None:
            self.logger.info("First deflation call in adaptiveHB finished.")
        
        hbCoeffList, _ = self.splitCoeffVec(self.deflationSolutionList[self.deflSolutionSelect])
        
        hbAmpRatioList = self.calcRelativeAmplitude(hbCoeffList)
        
        if self.nonAdaptive:
            self.previousResults.append([self.deflationSolutionList[self.deflSolutionSelect], \
                                        self.harmonicSetList])
            return self.deflationSolutionList[self.deflSolutionSelect], \
                self.harmonicSetList, [hbAmpRatioList]

        ################ adaptive part ##############
        
        # pop selected solution and perfom adaptive HB
        solCoeff = self.deflationSolutionList[self.deflSolutionSelect]
        self.deflationSolutionList.pop(self.deflSolutionSelect)
        self.deflationHarmonicSetList.pop(self.deflSolutionSelect)
        
        oldGamma = self.peakThreshold
        curIter = 0
        reductionIter = 0
        previousIndexSetListList = [self.harmonicSetList]
        hbAmpRatioListList = [hbAmpRatioList]
        self.adaptiveIterationSolList = [[solCoeff,self.harmonicSetList]]
        while max([len(x) for x in self.harmonicSetList]) < self.maxH \
            and curIter < self.maxIter:

            self.itrCt = 0

            if not curIter == 0:
                solCoeff = self.solveRootProblem(initialGuess)

                self.adaptiveIterationSolList.append([solCoeff,self.harmonicSetList])

            hbCoeffListSort, omega = self.splitCoeffVec(solCoeff)
                    
            posList = []
            velList = []
            for idxVar, hbCoeff in enumerate(hbCoeffListSort):
                pos, vel, _ = self.calculatePositionVelocityAcceleration(hbCoeff,
                                                        omega,
                                                        idxVar)
                posList.append(pos)
                velList.append(vel)
            
            f_ne_list = self.calculateNonlinearExternalForce(posList,
                                                            velList)
            
            initialGuessNew, emptyAddSet, newIndexSetList, \
                = self.updateIndexSet(f_ne_list,
                                      omega,
                                      hbCoeffListSort)
            if not curIter == 0:
                hbAmpRatioList = self.calcRelativeAmplitude(hbCoeffListSort)
                hbAmpRatioListList.append(hbAmpRatioList)

            if self.logger is not None:
                self.logger.info("New Harmonic set determined")

            if reductionIter > 10:
                break

            # check if refinement meets defined threshold
            if emptyAddSet == len(self.harmonicSetList):
                if any(hbAmpRatioList > self.ampThreshold):
                    self.peakThreshold = self.peakThreshold * 1e-1
                    self.harmonicSetList = newIndexSetList
                    
                    initialGuess = initialGuessNew
                    reductionIter += 1
                    curIter += 1
                    continue
                else:
                    break

            # supress oscillating between two sets
            if len(previousIndexSetListList) > 2:
                if newIndexSetList in previousIndexSetListList:
                    if any(hbAmpRatioList > self.ampThreshold):
                        self.peakThreshold = self.peakThreshold * 1e-1
                        self.harmonicSetList = newIndexSetList
                        
                        initialGuess = initialGuessNew
                        reductionIter += 1
                        curIter += 1
                        continue
                    else:
                        break
                else:
                    previousIndexSetListList.pop(0)
                    previousIndexSetListList.append(newIndexSetList)
            else:
                previousIndexSetListList.append(newIndexSetList)
            
            self.harmonicSetList = newIndexSetList
            initialGuess = initialGuessNew
            curIter += 1
            
            if curIter >= self.maxIter:
                if self.logger is not None:
                    self.logger.error("The maximum number of iterations has been reached.")
                raise ValueError("The maximum number of iterations has been reached.")

        # Reset gamma factor for new run
        self.peakThreshold = oldGamma
        self.firstRefinement = None
        
        if max([len(x) for x in self.harmonicSetList]) >= self.maxH:
            if self.logger is not None:
                self.logger.error("The maximum number of harmonics has been reached.")
            raise ValueError("The maximum number of harmonics has been reached.")
        if curIter >= self.maxIter:
            if self.logger is not None:
                self.logger.error("The maximum number of iterations has been reached.")
            raise ValueError("The maximum number of iterations has been reached.")
        
        if self.logger is not None:
            self.logger.info("adaptoveHB finished")

        self.previousResults.append([solCoeff, self.harmonicSetList])
        return solCoeff, self.harmonicSetList, hbAmpRatioListList

    def calculateDeflationsSolutions(self,
                                     initialGuess):
        r"""
        Calculate solutions and if deflation is active sort 
        the solutions after maximum amplitude

        Parameters
        ----------
        initialGuess : np.ndarray or list
            The initial guess for the solution.

        """
        
        if isinstance(initialGuess,np.ndarray):
            initialGuess = [initialGuess] 
        
        if self.deflationFakeSolutions is None:
            self.deflationSolutionList = []
            self.deflationHarmonicSetList = []
            maxPosValList = []
        else:
            if self.deflSolutionSelect != -1:
                raise ValueError("Deflation: For fake solutions the last entry of the list should be selected.")
            
            self.deflationSolutionList = copy.deepcopy(self.deflationFakeSolutions[0])
            self.deflationHarmonicSetList = copy.deepcopy(self.deflationFakeSolutions[1])
            maxPosValList = [0 for _ in self.deflationSolutionList]
    
        for idxDeflation in range(self.maxDefSol):

            if idxDeflation > len(initialGuess)-1:
                curInitialGuess = initialGuess[-1]
            else:
                curInitialGuess = initialGuess[idxDeflation]
            
            solCoeffDeflation = self.solveRootProblem(curInitialGuess)
            
            # determine max value for later sorting, 
            hbCoeffList, omega = self.splitCoeffVec(solCoeffDeflation)
            posMaxVarList = []
            for idxVar, hbCoeffVar in enumerate(hbCoeffList):
                pos, _, _ = self.calculatePositionVelocityAcceleration(hbCoeffVar,
                                                                        omega,
                                                                        idxVar)
                posMaxVarList.append(np.max(pos))
            
            # if there are multiple variables, take the max of the max values
            maxPosValList.append(max(posMaxVarList))
                
            self.deflationSolutionList.append(solCoeffDeflation)
            self.deflationHarmonicSetList.append(self.harmonicSetList)
        
        if self.deflationFakeSolutions is not None:
            # update maxPosVarList
            posMaxSol = maxPosValList[-1]
            scaleFac = np.linspace(0,posMaxSol,len(maxPosValList))
            maxPosValList = list(scaleFac)
            
        combinedList = sorted(zip(maxPosValList, 
                                  self.deflationSolutionList,
                                  self.deflationHarmonicSetList), key=lambda x: x[0])

        # Unzip to get sorted lists
        maxPosValListSorted, solCoeffDeflationListSorted, harmonicSetListListSorted = zip(*combinedList)
        maxPosValListSorted = list(maxPosValListSorted)
        self.deflationSolutionList = list(solCoeffDeflationListSorted)
        self.deflationHarmonicSetList = list(harmonicSetListListSorted)
        self.harmonicSetList = self.deflationHarmonicSetList[self.deflSolutionSelect]

    def calcRelativeAmplitude(self,
                              hbCoeffList: list):
        r"""
        This method calculates the relative amplitude of the harmonics in order
        to determine the convergence of the adaptive harmonic balance algorithm.

        Parameters
        ----------
        hbCoeffList : list
            List of harmonic coefficients for each variable.

        Returns
        ----------
        hbAmpRatioList : list
            List of the amplitude ratio for each variable
        """

        hbAmpRatioList = np.zeros(len(hbCoeffList))
        for idxVar, hbCoeff in enumerate(hbCoeffList):

            harmonicVarSet = self.harmonicSetList[idxVar]

            relevantCoeffList = []
            idxCoeffVar = 1
            for h in harmonicVarSet:
                relAmp = np.sqrt(hbCoeff[idxCoeffVar]**2+hbCoeff[idxCoeffVar+1]**2)
                if relAmp > np.finfo(float).eps:
                    relevantCoeffList.append(relAmp)
                idxCoeffVar += 2
            if relevantCoeffList == []:
                relevantCoeffList = [0]
            ampVec = np.array(relevantCoeffList)
            nrH = len(relevantCoeffList)

            if nrH > 1:
                if self.firstAHB[idxVar]:
                    self.firstAHB[idxVar] = False
                    self.maxAmp[idxVar] = np.max(ampVec)
                if self.maxAmp[idxVar] < np.max(ampVec):
                    self.maxAmp[idxVar] = np.max(ampVec)
                if self.maxAmp[idxVar] < np.finfo(float).eps or np.min(ampVec)==0:
                    hbAmpRatio = self.ampThreshold/len(self.maxAmp)
                else:
                    hbAmpRatio = np.min(ampVec)/self.maxAmp[idxVar]
            else:
                ampVec = np.sqrt(hbCoeff[1]**2+hbCoeff[2]**2)
                if ampVec < 1e-12:#np.finfo(float).eps:
                    ampVec = 0
                if self.firstAHB[idxVar]:
                    self.firstAHB[idxVar] = False
                    self.maxAmp[idxVar] = np.max(ampVec)
                    if ampVec == 0:
                        hbAmpRatio = 0
                    else:    
                        hbAmpRatio = self.ampThreshold/len(self.maxAmp)
                else:
                    if self.maxAmp[idxVar] < np.max(ampVec):
                        self.maxAmp[idxVar] = np.max(ampVec)
                    if self.maxAmp[idxVar] < np.finfo(float).eps:
                        hbAmpRatio = 0
                    else:    
                        hbAmpRatio = self.ampThreshold/len(self.maxAmp)
            hbAmpRatioList[idxVar] = hbAmpRatio
        
        return hbAmpRatioList
        
    def updateIndexSet(self,
                       f_ne_list: list,
                       omega: float,
                       hbCoeffList: list):
        r"""
        This method creates the updated index set

        Parameters
        ----------
        f_ne_list : list
            List of the time evaluation of the nonlinear excitation function.
        omega : float
            Base frequency of the system
        hbCoeffList : list
            List of harmonic coefficients for each variable.

        Returns
        ----------
        initialGuess : np.ndarray
            New inital guess for the updated index set
        emptyAddSet : int
            indicates how man variables have emtpy sets. If it matches the number of variables, refinement complet
        newHarmonicSetList : list
            list with the updated harmonic sets for each variable
        """
        
        # run index set before so that if for all variables the same harmonics can be used
        newHarmonicSetPreCheckList = []
        coeffSetPreCheckList = []
        for f_ne, solCoeff in zip(f_ne_list,
                                  hbCoeffList):

            newHarmonicSet, coeffNewSet\
                 = self.determineIndexSet(f_ne,
                                          omega,
                                          solCoeff)
            newHarmonicSetPreCheckList.append(newHarmonicSet)
            coeffSetPreCheckList.append(coeffNewSet)

        if self.sameH4all:
            unionHarmonicSet = set().union(*newHarmonicSetPreCheckList)
            newHSetadapted = []
            coeffSetSameHList = []
            for idx, preCheckHSet in enumerate(newHarmonicSetPreCheckList):
                oldHSet = self.harmonicSetList[idx]
                oldCoeffs = hbCoeffList[idx]
                preCheckCoeffs = coeffSetPreCheckList[idx]
                coeffSetSameH = np.empty(0)#array(oldCoeffs[0])[np.newaxis]

                idxOld = 1
                idxPreCheck = 0
                for curH in range(1,int(max(unionHarmonicSet)+1)):
                    if not self.forced:
                        print("Just make sure this is correct here!")
                        raise ValueError("Need to be checked")
                    nrCoeffs = 2
                    if curH in oldHSet and curH in unionHarmonicSet:
                        coeffSetSameH = np.concatenate([coeffSetSameH,oldCoeffs[idxOld:idxOld+nrCoeffs]], axis = 0)
                    elif curH in preCheckHSet:
                        coeffSetSameH = np.concatenate([coeffSetSameH,preCheckCoeffs[idxPreCheck:idxPreCheck+nrCoeffs]], axis = 0)
                    elif curH in unionHarmonicSet:
                        coeffSetSameH = np.concatenate([coeffSetSameH,np.zeros(nrCoeffs)], axis = 0)

                    if curH in oldHSet:
                        idxOld +=nrCoeffs
                    if curH in preCheckHSet:
                        idxPreCheck +=nrCoeffs

                newHSetadapted.append(unionHarmonicSet)
                coeffSetSameHList.append(coeffSetSameH)

            determinedHarmonicSetList = newHSetadapted
            coeffNewSetList = coeffSetSameHList

        else:
            determinedHarmonicSetList = copy.deepcopy(newHarmonicSetPreCheckList)
            coeffNewSetList = coeffSetPreCheckList
        
        emptyAddSet = 0
        idxVar1 = 0
        newHarmonicSetList = []
        initialCoeffsVarList = []
        for f_ne, oldHarmonicSet, solCoeff, newHarmonicSet, coeffNewSet in zip(f_ne_list,
                                                  self.harmonicSetList,
                                                  hbCoeffList,
                                                  determinedHarmonicSetList,
                                                  coeffNewSetList):

            
            if len(newHarmonicSet) == 0 :
                newHarmonicSet = oldHarmonicSet
        
            # create sets
            addSet = newHarmonicSet - oldHarmonicSet
            if self.onlyAdd:
                removeSet = {}
            else:
                removeSet = oldHarmonicSet - newHarmonicSet

            if idxVar1 == 0 and not self.forced:
                if self.onlyAdd:
                    nrCoeffs = len(newHarmonicSet.union(oldHarmonicSet))*2
                else:
                    nrCoeffs = len(newHarmonicSet)*2
                solCoeffVar = np.concatenate([solCoeff[:2],solCoeff[3:]])
                coeffNewSetVar = np.concatenate([coeffNewSet[:1],coeffNewSet[2:]])
            else:
                if self.onlyAdd:
                    nrCoeffs = len(newHarmonicSet.union(oldHarmonicSet))*2+1
                else:
                    nrCoeffs = len(newHarmonicSet)*2+1
                solCoeffVar = solCoeff
                coeffNewSetVar = coeffNewSet

            if not addSet and not removeSet:
                emptyAddSet += 1

                if not removeSet:
                    initialCoeffsVarList.append(solCoeffVar)
                else:
                    initialCoeffsVarList.append(solCoeffVar[:nrCoeffs])
            else:

                initialCoeffsVar = np.zeros(nrCoeffs)
                initialCoeffsVar[0] = solCoeff[0]
                idxOld = 1
                
                idxNew = 1
                idxNewOnly = 1
                firstHarmonic = True
                for h in newHarmonicSet.union(oldHarmonicSet):
                    
                    if not self.forced and firstHarmonic and idxVar1 == 0:
                        addCoeff = 1
                        firstHarmonic = False
                    else:
                        addCoeff = 2

                    if h in oldHarmonicSet:
                        if h in newHarmonicSet or self.onlyAdd:
                            initialCoeffsVar[idxNew:idxNew+addCoeff] = solCoeffVar[idxOld:idxOld+addCoeff]
                            idxNew += addCoeff
                            if h in newHarmonicSet:
                                idxNewOnly += addCoeff
                        idxOld += addCoeff
                    elif h in newHarmonicSet:
                        initialCoeffsVar[idxNew:idxNew+addCoeff] = coeffNewSetVar[idxNewOnly-1:idxNewOnly-1+addCoeff]
                        idxNew += addCoeff
                
                initialCoeffsVarList.append(initialCoeffsVar)
            if self.onlyAdd:
                newHarmonicSetList.append(newHarmonicSet.union(oldHarmonicSet))
            else:
                newHarmonicSetList.append(newHarmonicSet)
            idxVar1 += 1

        initialGuess = np.concatenate(initialCoeffsVarList)

        if not self.forced:
            initialGuess = np.append(initialGuess, omega)
        
        return initialGuess, emptyAddSet, newHarmonicSetList
    
    def residualFunctionDeflation(self,
                                  currentCoeffs: np.ndarray):
        r"""
        This method deflates the algorithm, when selected

        Parameters
        ----------
        currentCoeffs : np.ndarray
            Current coefficients of the harmonic balance solution.

        Returns
        ----------
        residual : np.ndarray
            The residual of the harmonic balance solution.
        """

        residual = self.residualFunction(currentCoeffs)

        if self.deflation:
            deflationOperator = self.getDeflationOperator(currentCoeffs)
            residual = np.dot(deflationOperator, residual)

        return residual

    def getDeflationOperator(self,
                             x: np.ndarray):
        r"""
        Method returns the deflation operator for current guess

        Parameters
        ----------
        x : np.ndarray
            current guess of coefficients

        Returns
        ----------
        deflationOperator : np.ndarray
            deflation operator for the current guess
        """
        if len(self.deflationSolutionList) == 0:
            deflationOperator = np.identity(x.shape[0]) 
        else:
            i = 0
            for sol, harmSet in zip(self.deflationSolutionList,self.deflationHarmonicSetList):
                # adapt solution to current HB
                if sol.shape[0] != x.shape[0]:

                    xCoeffList, xOmega = self.splitCoeffVec(x)

                    idxSolVarH = 0

                    diffVecList = []
                    # sol and x have same number of dofs
                    for idxVar, solVarHSet in enumerate(harmSet):
                        if idxSolVarH == 0 and not self.forced:
                            solLenH = len(solVarHSet)*2
                            xCoeffsVarFull = xCoeffList[idxVar]
                            xCoeffsVar = np.concatenate([xCoeffsVarFull[:2],xCoeffsVarFull[3:]], axis=0)
                        else:
                            solLenH = len(solVarHSet)*2+1
                            xCoeffsVar = xCoeffList[idxVar]
                        solCoeffsVar = sol[idxSolVarH:idxSolVarH+solLenH]

                        
                        xVarHSet = self.harmonicSetList[idxVar]

                        interSectSet = solVarHSet.intersection(xVarHSet)
                        
                        if idxSolVarH == 0 and not self.forced:
                            solExtractVec = np.zeros(len(interSectSet)*2-1)
                        else:
                            solExtractVec = np.zeros(len(interSectSet)*2)
                        xExtractVec = np.zeros_like(solExtractVec)

                        # extract sol Coeffs
                        def extractCoeffs(varHSet,extractVec,coeffsVar,interSectSet):
                            idxEx = 0
                            idxCoeff = 1
                            for curH in varHSet:
                                if curH in interSectSet:
                                    if idxSolVarH == 0 and not self.forced:
                                        extractVec[idxEx:idxEx+1] = coeffsVar[idxCoeff:idxCoeff+1]
                                        idxEx += 1
                                    else:
                                        extractVec[idxEx:idxEx+2] = coeffsVar[idxCoeff:idxCoeff+2]
                                        idxEx += 2
                                if idxSolVarH == 0 and not self.forced:
                                    idxCoeff += 1
                                else:
                                    idxCoeff += 2
                            return extractVec

                        solExtractVec = extractCoeffs(solVarHSet,solExtractVec,solCoeffsVar,interSectSet)
                        xExtractVec = extractCoeffs(xVarHSet,xExtractVec,xCoeffsVar,interSectSet)

                        # exclude Zeros from deflation
                        maskSol = np.abs(solExtractVec) > 1e-15
                        maskX = np.abs(xExtractVec) > 1e-15
                        maskDiff = np.logical_or(maskSol, maskX)
                        
                        if all(np.invert(maskDiff)):
                            diffVec = np.array([xCoeffsVar[0]-solCoeffsVar[0]])
                        else:
                            diffVec = np.concatenate([np.array([xCoeffsVar[0]-solCoeffsVar[0]]), xExtractVec[maskDiff] - solExtractVec[maskDiff]], axis=0)

                        diffVecList.append(diffVec)
                        idxSolVarH += solLenH

                    diffVectotal = np.concatenate(diffVecList,axis=0)

                    if not self.forced:
                        diffVectotal = np.concatenate([diffVectotal,xOmega-sol[-1]])
                    
                    normVal = np.linalg.norm(diffVectotal)

                else:
                    normVal = np.linalg.norm(x-sol)
                
                identityMat = np.identity(x.shape[0])
                deflationOpera = identityMat/(normVal**self.defl_p)+self.defl_alpha*identityMat
                if i==0:
                    deflationOperator = deflationOpera
                    i = 1
                else:
                    deflationOperator = np.matmul(deflationOperator,deflationOpera)
        
        return deflationOperator
    
    def calcPreviousSolutionInitialGuess(self):
        r"""
        Function to calculate initial guesses for the current parameter set, based on 
        previous solutions. Updated only to current harmonics.

        Returns
        ----------
        newInitialGuessList : list
            list with new initial guesses for the root-finding algorithm.
        """
        newInitialGuessList = []
        for previousSol in self.previousResults:

            previousFourierCoeffVec = previousSol[0]
            previousHarmonicSet = previousSol[1]

            newFourierList = []
            idxPrev = 0
            for idxVar, previousHarmVar in enumerate(previousHarmonicSet):
                curHVar = self.harmonicSetList[idxVar]
                if idxVar == 0 and not self.forced:
                    idxPrevFourierCoeff = len(previousHarmVar)*2
                    idxNewFourierCoeff = len(curHVar)*2
                else:
                    idxPrevFourierCoeff = len(previousHarmVar)*2+1
                    idxNewFourierCoeff = len(curHVar)*2+1

                previousFourierCoeffVecVar = previousFourierCoeffVec[idxPrev:idxPrev+idxPrevFourierCoeff]
                
                if idxNewFourierCoeff <= idxPrevFourierCoeff:
                    newFourierVecVar = previousFourierCoeffVecVar[:idxNewFourierCoeff]
                else:
                    newFourierVecVar = np.zeros(idxNewFourierCoeff)
                    newFourierVecVar[:idxPrevFourierCoeff] = previousFourierCoeffVecVar

                newFourierList.append(newFourierVecVar)

                idxPrev += idxPrevFourierCoeff

            if not self.forced:
                newFourierList.append(np.array(previousFourierCoeffVec[-1]))

            newInitialGuessList.append(np.hstack(newFourierList))
        
        return newInitialGuessList
