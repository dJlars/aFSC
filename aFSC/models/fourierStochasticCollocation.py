r"""
Author:         Lars de Jong
Date:           2024-12-20
Description:    class handles the creation of a 
                Fourier stochastic collocation surrogate model.
"""
import copy
import logging
import numpy as np

from typing import Union

from aFSC.models.adaptiveHB import adaptiveHarmonicBalance
from aFSC.utils.helperClasses import uncertainDimension

from aFSC.utils.idx_admissibility import is_admissible, \
    admissible_neighborhood, update_active_set

class fourierStochasticCollocation:
    r"""
    Class handles the creation of a Fourier stochastic collocation 
    surrogate model. Among other functions, this includes expecially
    the creation of collocation points, the function evaluation at 
    those points, determination of admissible neighbor indices, 
    the calculation of hierarchical surpluses and the adaptive
    refinement of the surrogate model.
    """

    def __init__(self,
                 distList: list,
                 adaptiveHB: adaptiveHarmonicBalance,
                 logger: logging.Logger = None,
                 **kwargs):
        r"""
        Parameters
        ----------
        distList : list
            List of chaospy distributions for the uncertain dimensions.
        adaptiveHB : adaptiveHarmonicBalance
            Object of the adaptive harmonic balance class

        **kwargs : dict
            Optional parameters for the Fourier stochastic collocation model.
            - max_fcalls : int, optional
                Maximum number of function calls for the adaptive refinement.
                Default is 1000.
            - fscError : float, optional
                Error threshold for the stochastic collocation model.
                Default is 1e-3.
            - scHBBalFac : float, optional
                Balance factor between stochastic collocation and harmonic balance error.
                Default is 0.5.
            - nonAdaptive : bool, optional
                If True, the model will not adaptively refine itself.
                Default is False.
            - maxCP : int, optional
                Maximum number of collocation points to be used in the model.
                Default is 1000.
        """
        self.logger = logger

        self.d = len(distList) # number of uncertain dimensions
        self.distList = distList

        self.uncertainDimList = [uncertainDimension(dist) for dist in distList]
        
        self.adaptiveHB = adaptiveHB

        self.stochasticCollocationModel = []
        self.nrSCModels = 0

        self.maxHarmonicSet = [{1} for i in range(self.adaptiveHB.nrVar)]

        self.indexSetList = []
        self.activeIndexSetList = []
        self.collocationPointList = []
        self.fourierCoeffList = []
        self.harmonicSetListList = []
        self.multiLagrangeList = []
        self.hierarchicalSurpluseList = []
        self.hierarchicalSurplusHarmSetList = []
        self.errorHBEstimatorperCPList = []
        self.errorSCAdmissibleList = []
        self.errorSCList = []
        self.errorHBList = []

        self.firstHSList = None
        self.firstScalHS = None

        # optional parameters
        self.max_fcalls = kwargs.get("max_fcalls", 1000)

        self.fscError = kwargs.get("fscError", 1e-3)
        self.scBalanceFactor = kwargs.get("scHBBalFac", 0.5)

        self.scTol = self.scBalanceFactor * self.fscError
        self.hbTol = (1-self.scBalanceFactor) * self.fscError

        self.nonAdaptive = kwargs.get("nonAdaptive", False)
        self.maxCP = kwargs.get("maxCP", 1000)

        self.resetHB = kwargs.get("resetHB",False)

        self.meanAsStart = kwargs.get("meanAsStart", False)

        self.defHBThreshold = None

        if self.logger is not None:
            self.logger.info("aFSC Model created!")


    def calcFSCSurrogate(self,
                         initialGuess: Union[int, float, list, np.ndarray],
                         initialMultiIndexSet: list,
                         activeSet: list):
        r"""
        This function creates the Fourier stochastic collocation
        surrogate model.

        Parameters
        ----------
        initialGuess : Union[int, float, np.ndarray]
            The initial guess for the Fourier coefficients.
        initialMultiIndexSet : list
            List of multi-indices for the initial collocation points.
        activeSet : list
            List of active multi-indices of the initialMultiIndexSet.
        """
        
        if isinstance(initialGuess, (int, float, str, list)):
            initialGuessVec = self.adaptiveHB.getInitialGuess(initialGuess)
        elif isinstance(initialGuess, np.ndarray):
            initialGuessVec = initialGuess
        else:
            raise ValueError("Initial guess must be a scalar or a numpy array.")
        
        for initialMultiIndex in initialMultiIndexSet:
            
            # check for downward closure
            if not is_admissible(initialMultiIndex, self.indexSetList):
                raise ValueError(f"""Multi index {initialMultiIndex} is not admissible. 
                                 Index set is not downward closed.""")
            
            if self.resetHB:
                originalHBSet = copy.deepcopy(self.adaptiveHB.harmonicSetList)

            initialCollocationPoint, initialHBCoeff, \
                initialHarmonicSet, initialHierarchicalSurpluse, initialHBErrorEstimatorListList, \
                harmonicSetHS \
                      = self.calcNextSCSurrogateParts(initialMultiIndex, initialGuessVec)

            multiLagrange = self.getMultivariateLagrange(initialMultiIndex)
        
            # store SC model parts
            self.indexSetList.append(initialMultiIndex)
            self.collocationPointList.append(initialCollocationPoint)
            self.fourierCoeffList.append(initialHBCoeff)
            self.appendHarmonicSetList(initialHarmonicSet)
            self.hierarchicalSurpluseList.append(initialHierarchicalSurpluse)
            self.hierarchicalSurplusHarmSetList.append(harmonicSetHS)
            self.multiLagrangeList.append(multiLagrange)
            self.errorHBEstimatorperCPList.append(initialHBErrorEstimatorListList[-1])

            if initialMultiIndex == initialMultiIndexSet[0]:
                self.errorSCList.append(sum(self.calcHierarchicalError(initialHierarchicalSurpluse,initialHarmonicSet))/self.adaptiveHB.nrVar)

            self.errorHBList.append(0.5*sum([addAmp**2 if not all(addAmp == self.adaptiveHB.ampThreshold/len(self.adaptiveHB.maxAmp)) else self.adaptiveHB.ampThreshold/len(self.adaptiveHB.maxAmp) for addAmp in self.errorHBEstimatorperCPList]))
            self.nrSCModels += 1
                    
            if self.meanAsStart:
                initialGuessVec = self.adaptiveHB.getInitialGuess(initialGuess)
            else:
                initialGuessVec = initialHBCoeff
                    
            if self.resetHB:
                self.adaptiveHB.harmonicSetList = originalHBSet
                initialGuessVec = self.adaptiveHB.getInitialGuess(initialGuess)

            if self.logger is not None:
                self.logger.info(f"Mulit index {initialMultiIndex}")

        if self.logger is not None:
            self.logger.info("Initial multi index set finished!")

        fcalls = 0
        adm_IndexSet = []
        adm_CollocationPointList = []
        adm_FourierCoeffList = []
        adm_HarmonicSetList = []
        adm_HierarchicalSurpluseList = []
        adm_hierarchicalSurplusHarmsetList = []
        adm_MultiLagrangeList = []
        adm_errorHBperCPList = []
        adm_errorSCList = []
        SC_error = self.errorSCList[-1]
        
        while SC_error > self.scTol and fcalls < self.max_fcalls and len(self.collocationPointList) < self.maxCP:
            adm_IndexSet, activeSet, add_adm_IndexSet = \
                admissible_neighborhood(self.indexSetList, activeSet, adm_IndexSet)
            
            # loop through new admissible sets
            for admIndex in add_adm_IndexSet:

                if admIndex == [5,0]:
                    print("Hi")

                if self.resetHB:
                    originalHBSet = copy.deepcopy(self.adaptiveHB.harmonicSetList)

                collocationPoint, fourierCoeffVec, \
                    harmonicSet, hierarchicalSurpluse, \
                        hbErrorEstimatorListList, harmonicSetHS, \
                              = self.calcNextSCSurrogateParts(admIndex, 
                                                              initialGuessVec)
                
                multiLagrange = self.getMultivariateLagrange(admIndex)

                # store admissible SC model parts
                adm_CollocationPointList.append(collocationPoint)
                adm_FourierCoeffList.append(fourierCoeffVec)
                adm_HarmonicSetList.append(copy.deepcopy(harmonicSet))
                adm_HierarchicalSurpluseList.append(hierarchicalSurpluse)
                adm_hierarchicalSurplusHarmsetList.append(harmonicSetHS)
                adm_MultiLagrangeList.append(multiLagrange)
                adm_errorHBperCPList.append(hbErrorEstimatorListList)
                adm_errorSCList.append(self.calcHierarchicalError(hierarchicalSurpluse,harmonicSetHS))
                if self.meanAsStart:
                    initialGuessVec = self.adaptiveHB.getInitialGuess(initialGuess)
                else:
                    initialGuessVec = fourierCoeffVec
                fcalls += 1

                if self.resetHB:
                    self.adaptiveHB.harmonicSetList = originalHBSet
                    initialGuessVec = self.adaptiveHB.getInitialGuess(initialGuess)
                if self.logger is not None:
                    self.logger.info(f"Admissible Mulit index {admIndex} calculated!")

            adm_totalErrorList = []
            for cpErrors in adm_errorSCList:
                sumCPErrors = sum(cpErrors)/self.adaptiveHB.nrVar
                if sumCPErrors == 0:
                    sumCPErrors = self.scTol/len(adm_errorSCList)
                adm_totalErrorList.append(sumCPErrors)
            self.errorSCAdmissibleList.append(copy.deepcopy(adm_errorSCList))
            
            # check for greatest contribution to error
            add_adm_Index = np.argmax(adm_totalErrorList)
            SC_error = sum(adm_totalErrorList)

            if self.logger is not None:
                self.logger.info(f"Nr used CP {len(self.indexSetList)}. Added index {adm_IndexSet[add_adm_Index]}")
            
            # add the admissible index with the greatest contribution to active index set
            self.indexSetList.append(adm_IndexSet[add_adm_Index])
            activeSet.append(adm_IndexSet[add_adm_Index])
            activeSet = update_active_set(activeSet)
            self.activeIndexSetList.append(copy.deepcopy(activeSet))
            self.collocationPointList.append(adm_CollocationPointList[add_adm_Index])
            self.fourierCoeffList.append(adm_FourierCoeffList[add_adm_Index])
            self.appendHarmonicSetList(adm_HarmonicSetList[add_adm_Index])
            self.hierarchicalSurpluseList.append(adm_HierarchicalSurpluseList[add_adm_Index])
            self.hierarchicalSurplusHarmSetList.append(adm_hierarchicalSurplusHarmsetList[add_adm_Index])
            self.multiLagrangeList.append(adm_MultiLagrangeList[add_adm_Index])
            self.errorHBEstimatorperCPList.append(adm_errorHBperCPList[add_adm_Index][-1])
            self.errorSCList.append(SC_error)
            self.nrSCModels += 1

            print(f"\nNr used CP {len(self.indexSetList)}. Added index {adm_IndexSet[add_adm_Index]}")
            
            # remove added entries from admissible index set
            adm_IndexSet.pop(add_adm_Index)
            adm_CollocationPointList.pop(add_adm_Index)
            adm_FourierCoeffList.pop(add_adm_Index)
            adm_HarmonicSetList.pop(add_adm_Index)
            adm_HierarchicalSurpluseList.pop(add_adm_Index)
            adm_hierarchicalSurplusHarmsetList.pop(add_adm_Index)
            adm_MultiLagrangeList.pop(add_adm_Index)
            adm_errorHBperCPList.pop(add_adm_Index)
            adm_errorSCList.pop(add_adm_Index)

            adm_errorSCListNew = []
            adm_HierarchicalSurpluseListNew = []
            for idxAdmCP in range(len(adm_IndexSet)):
                newHS, \
                newHarmonicSetHS = self.getHierarchicalSurpluse(adm_FourierCoeffList[idxAdmCP],
                                                                adm_CollocationPointList[idxAdmCP],
                                                                adm_HarmonicSetList[idxAdmCP])
                
                adm_errorSCListNew.append(self.calcHierarchicalError(newHS,newHarmonicSetHS))
                adm_HierarchicalSurpluseListNew.append(newHS)

            
            self.errorHBList.append(0.5*sum([addAmp**2 if not all(addAmp == self.adaptiveHB.ampThreshold/len(self.adaptiveHB.maxAmp)) else self.adaptiveHB.ampThreshold/len(self.adaptiveHB.maxAmp) for addAmp in self.errorHBEstimatorperCPList]))
            
            print(f"Current HB error: [{np.sum(self.errorHBList):.3e}], SC error: {SC_error:.3e}")
            if self.logger is not None:
                self.logger.info(f"Current HB error: [{np.sum(self.errorHBList):.3e}], SC error: {SC_error:.3e}")
            
            if len(self.collocationPointList) >= self.maxCP:
                break

            if self.nonAdaptive:
                break
            
        print(f"\nSC surrogate model created. HB error: [{np.sum(self.errorHBList[-1]):.3e}], SC error: {SC_error:.3e}")


        # add all already calculated SC models to the surrogate model
        if len(self.collocationPointList) < self.maxCP:
            self.addedAdmissibleSet = adm_IndexSet
            self.addedAdmissibleCP = adm_CollocationPointList
            self.indexSetList += adm_IndexSet
            self.collocationPointList += adm_CollocationPointList
            self.fourierCoeffList += adm_FourierCoeffList
            for admHarmSet in adm_HarmonicSetList:
                self.appendHarmonicSetList(admHarmSet)
            self.hierarchicalSurpluseList += adm_HierarchicalSurpluseList
            self.hierarchicalSurplusHarmSetList += adm_hierarchicalSurplusHarmsetList
            self.multiLagrangeList += adm_MultiLagrangeList
            self.errorHBEstimatorperCPList += adm_errorHBperCPList
            self.nrSCModels += len(adm_IndexSet)
            self.adaptiveHB.harmonicSetList = self.maxHarmonicSet

    def getSparseGridPoint(self,
                           multiIndex: list):
        r"""
        Returns the sparse grid point for a given multi-index.

        Parameters
        ----------
        multiIndex : list
            The multi-index for which the sparse grid point is requested.

        Returns
        ----------
        sparseGridPoint : list
            The sparse grid point for the given multi-index.
        """
        sparseGridPoint = []
        for i in range(self.d):
            dimPoint = self.uncertainDimList[i].getKnot(multiIndex[i])
            sparseGridPoint.append(dimPoint)
        
        return sparseGridPoint
    
    def calcNextSCSurrogateParts(self,
                                 multiIndex: list,
                                 initialGuessVec: np.ndarray):
        r"""
        This method calculates the aHB for the given CP and its hierarchical surplus.

        Parameters
        ----------
        multiIndex : list
            The multi-index for which the aHB is calculated.
        initialGuessVec : np.ndarray
            The initial guess for the Fourier coefficients.

        Returns
        ----------
        collocationPoint : np.ndarray
            The collocation point for the given multi-index.
        fourierCoeffVec : np.ndarray
            The Fourier coefficients for the collocation point.
        harmonicSet : list
            The harmonic set for the collocation point.
        hierarchicalSurpluse : np.ndarray
            The hierarchical surplus for the collocation point.
        hbErrorEstList : list
            The error estimator list for the harmonic balance method.
        harmonicSetHS : list
            The harmonic set for the collocation point for the hierarchical surplus.
        """
        
        collocationPoint = self.getSparseGridPoint(multiIndex)
        if self.defHBThreshold is None:
            self.defHBThreshold = np.sqrt(self.fscError/(2))
            self.adaptiveHB.ampThreshold = self.hbTol
        self.adaptiveHB.assignUQParameters(collocationPoint)

        fourierCoeffVec, harmonicSet, hbErrorEstList = self.adaptiveHB.adaptiveHB(initialGuessVec)

        hierarchicalSurpluse, \
            harmonicSetHS = self.getHierarchicalSurpluse(fourierCoeffVec,
                                                         collocationPoint,
                                                         harmonicSet)

        return collocationPoint, fourierCoeffVec, \
            harmonicSet, hierarchicalSurpluse, hbErrorEstList, \
            harmonicSetHS
    
    def getMultivariateLagrange(self,
                                multiIndex: list):
        r"""
        This method returns the multivariate Lagrange polynomials for the given multi-index.

        Parameters
        ----------
        multiIndex : list
            The multi-index for which the multivariate Lagrange polynomials are requested.
        
        Returns
        ----------
        multiLagrange : list
            The multivariate Lagrange polynomials for the given multi-index.
        """
        
        multiLagrange = []
        for idxDim, degree in enumerate(multiIndex):
            multiLagrange.append(self.uncertainDimList[idxDim].polynomials[degree])
        return multiLagrange
    
    def evalMultiLagrange(self,
                          multiLagrange: list,
                          sparseGridPoint: np.ndarray):
        r"""
        This method evaluates the multivariate Lagrange polynomials at the given sparse grid point.

        Parameters
        ----------
        multiLagrange : list
            The multivariate Lagrange polynomials for the given multi-index.
        sparseGridPoint : np.ndarray
            The sparse grid point at which the multivariate Lagrange polynomials are evaluated.

        Returns
        ----------
        mlEval : float
            The evaluation of the multivariate Lagrange polynomials at the given sparse grid point.
        """
        mlEval = 1
        for idxDim, lagrange in enumerate(multiLagrange):
            mlEval *= lagrange.evaluate(sparseGridPoint[idxDim])[0]
        return mlEval
    
    def stochasticCollocationEval(self,
                                  sparesGridPoint: np.ndarray):
        r"""
        This method evaluates the stochastic collocation surrogate model at the given sparse grid point.

        Parameters
        ----------
        sparesGridPoint : np.ndarray
            The sparse grid point at which the stochastic collocation surrogate model is evaluated.

        Returns
        ----------
        scModel : np.ndarray
            The evaluation of the stochastic collocation surrogate model at the given sparse grid point.
        """

        curAHBHarmSet = copy.deepcopy(self.adaptiveHB.harmonicSetList)
        
        for idxSCM, (curHSVec,curHSHarmSet) in enumerate(zip(self.hierarchicalSurpluseList,
                                                             self.hierarchicalSurplusHarmSetList)):
            
            self.adaptiveHB.harmonicSetList = curHSHarmSet
            curHSCoeffList, curOmega = self.adaptiveHB.splitCoeffVec(curHSVec)

            curHSCoeffFinalList = []
            for idxVar, (curHSCoeffVar, curHSHarm) in enumerate(zip(curHSCoeffList,
                                                                    curHSHarmSet)):

                curHSCoeffFinalVarList = [np.array([curHSCoeffVar[0]])]
                idxH = 1
                for h in self.maxHarmonicSet[idxVar]:
                    if h in curHSHarm:
                        curHSCoeffFinalVarList.append(curHSCoeffVar[idxH:idxH+2])
                        idxH+=2
                    else:
                        curHSCoeffFinalVarList.append(np.zeros(2))

                curHSCoeffFinalVar = np.concatenate(curHSCoeffFinalVarList)
                if not self.adaptiveHB.forced and idxVar == 0:
                    curHSCoeffFinalVar = np.concatenate((curHSCoeffFinalVar[:2],curHSCoeffFinalVar[3:]))

                curHSCoeffFinalList.append(curHSCoeffFinalVar)

            curHSCoeffFinal = np.concatenate(curHSCoeffFinalList)

            if not self.adaptiveHB.forced:
                curHSCoeffFinal = np.concatenate((curHSCoeffFinal,np.array([curOmega])))
            # Evaluate Polynomial
            multiLagEval = self.evalMultiLagrange(self.multiLagrangeList[idxSCM], 
                                                  sparesGridPoint)

            if idxSCM == 0:
                scModel = curHSCoeffFinal* multiLagEval
            else:
                scModel += curHSCoeffFinal* multiLagEval

        self.adaptiveHB.harmonicSetList = curAHBHarmSet
        return scModel
    
    def getHierarchicalSurpluse(self,
                                fourierCoeffVec: np.ndarray,
                                sparseGridPoint: np.ndarray,
                                harmonicSet: list):
        r"""
        This method calculates the hierarchical surplus for the given Fourier coefficients
        at the given sparse grid point.

        Parameters
        ----------
        fourierCoeffVec : np.ndarray
            The Fourier coefficients for the sparse grid point.
        sparseGridPoint : np.ndarray
            The sparse grid point at which the hierarchical surplus is calculated.
        harmonicSet : list
            The harmonic set for the sparse grid point.

        Returns
        ----------
        hierarchicalSurpluse : np.ndarray
            The hierarchical surplus for the given Fourier coefficients at the sparse grid point.
        harmonicSetHS : list
            The harmonic set for the collocation point for the hierarchical surplus.
        """

        if self.hierarchicalSurpluseList == []:
            hierarchicalSurpluse = fourierCoeffVec
            harmonicSetHS = harmonicSet
            inbetweenHarmonics = [{} for _ in harmonicSet]
        else:
            curHarmonicSet = self.adaptiveHB.harmonicSetList
            previousSCEval = self.stochasticCollocationEval(sparseGridPoint)

            self.adaptiveHB.harmonicSetList = self.maxHarmonicSet
            previousSCEvalList, prevOmega = self.adaptiveHB.splitCoeffVec(previousSCEval)

            self.adaptiveHB.harmonicSetList = harmonicSet
            fourierCoeffVecList, curOmega = self.adaptiveHB.splitCoeffVec(fourierCoeffVec)

            hierarchicalSurpluseList = []
            harmonicSetHS = []
            inbetweenHarmonics = []
            for idxVar, (prevSCEvalVar, 
                         fourierCoeffVar, 
                         maxHarmSetVar, 
                         harmSetVar) in enumerate(zip(previousSCEvalList,
                                                   fourierCoeffVecList,
                                                   self.maxHarmonicSet,
                                                   harmonicSet)):

                # check for mismatch of previousSC and fourierCoeffVec 
                if fourierCoeffVar.shape[0] < prevSCEvalVar.shape[0]:
                    
                    newFourierCoeffVecVar = np.zeros_like(prevSCEvalVar)
                    newFourierCoeffVecVar[0] = fourierCoeffVar[0]
                    
                    idxNewFC = 1
                    idxOldFC = 1
                    for currH in maxHarmSetVar:
                        
                        if currH in harmonicSet[idxVar]:
                            newFourierCoeffVecVar[idxNewFC:idxNewFC+2] = fourierCoeffVar[idxOldFC:idxOldFC+2]
                            idxOldFC += 2
                        else:
                            newFourierCoeffVecVar[idxNewFC:idxNewFC+2] = np.zeros(2)
                        idxNewFC += 2
                    
                    hierarchicalSurpluseVar = newFourierCoeffVecVar - prevSCEvalVar

                    harmonicSetHS.append(maxHarmSetVar)
                    inbetweenHarmonics.append({})

                # only new harmonics can be added, previous ones are kept
                elif  fourierCoeffVar.shape[0] > prevSCEvalVar.shape[0]:

                    newHarmonics = harmSetVar - maxHarmSetVar
                    inBetweenHarmonicsVar = {h for h in newHarmonics if h < max(maxHarmSetVar)}
                    if inBetweenHarmonicsVar:
                        newPreviousSCEvalList = [np.array([prevSCEvalVar[0]])]
                        idxSC = 1
                        for h in harmSetVar:
                            if h in maxHarmSetVar:
                                newPreviousSCEvalList.append(prevSCEvalVar[idxSC:idxSC+2])
                                idxSC += 2
                            else:
                                newPreviousSCEvalList.append(np.zeros(2))
                                 
                        newPreviousSCEval = np.concatenate(newPreviousSCEvalList)
                        
                    else:
                        newPreviousSCEval = np.zeros_like(fourierCoeffVar)
                        newPreviousSCEval[:prevSCEvalVar.shape[0]] = prevSCEvalVar

                    hierarchicalSurpluseVar = fourierCoeffVar - newPreviousSCEval
                    harmonicSetHS.append(harmSetVar)
                    inbetweenHarmonics.append(inBetweenHarmonicsVar)
                        
                else:
                    hierarchicalSurpluseVar = fourierCoeffVar - prevSCEvalVar
                    harmonicSetHS.append(maxHarmSetVar)
                    inbetweenHarmonics.append({})
                
                if not self.adaptiveHB.forced and idxVar == 0:
                    hierarchicalSurpluseVar = np.concatenate((hierarchicalSurpluseVar[:2], hierarchicalSurpluseVar[3:]))

                hierarchicalSurpluseList.append(hierarchicalSurpluseVar)
            
            hierarchicalSurpluse = np.concatenate(hierarchicalSurpluseList)

            self.adaptiveHB.harmonicSetList = curHarmonicSet

            if not self.adaptiveHB.forced:
                hsOmega = curOmega - prevOmega
                hierarchicalSurpluse = np.concatenate((hierarchicalSurpluse, np.array([hsOmega])))

        return hierarchicalSurpluse, harmonicSetHS
        
    def calcHierarchicalError(self,
                              hierarchicalSurpluse: np.ndarray,
                              harmonicSetHS: list):
        r"""
        This method calculates the error for the given hierarchical surplus.
        
        Parameters
        ----------
        hierarchicalSurpluse : np.ndarray
            The hierarchical surplus for which the error is calculated.
        harmonicSetHS : list
            The harmonic set for the collocation point for the hierarchical surplus.

        Returns
        ----------
        totalErrorHS : list
            The error for the hierarchical surplus for each variable.
        """
        
        curHarmset = copy.deepcopy(self.adaptiveHB.harmonicSetList)
        
        self.adaptiveHB.harmonicSetList = harmonicSetHS
        hierarchicalCoeffList, _ = self.adaptiveHB.splitCoeffVec(hierarchicalSurpluse)
        totalHSvalList = []
        firstHSList = []
        for idxVar, hierarchicalCoeffVarOrg in enumerate(hierarchicalCoeffList):
            
            # elimnate values less than 1e-8 of maximum value, if less than machin precision set to it
            hierarchicalCoeffVar = copy.deepcopy(hierarchicalCoeffVarOrg)
            max_coeff = np.max(np.abs(hierarchicalCoeffVar))
            threshold = max(max_coeff*1e-5,1e-12)
            hierarchicalCoeffVar[np.abs(hierarchicalCoeffVar) < threshold] = 0 

            if not self.adaptiveHB.forced and idxVar == 0:
                hsVar = np.concatenate((hierarchicalCoeffVar[0:2], hierarchicalCoeffVar[3:]))
            else:
                hsVar = hierarchicalCoeffVar
            squareHS = (hsVar)**2
            squareHS[1:] = squareHS[1:]/2
            totalHSval = np.sum(squareHS)

            if totalHSval < 1e-12:
                totalHSval = 0
            
            if self.firstHSList is None:
                if totalHSval == 0:
                    totalHSval = 0
                firstHSList.append(totalHSval)
                
            totalHSvalList.append(totalHSval)

        if self.firstHSList is None:
            firstHSListOrg = np.array(copy.deepcopy(firstHSList))
            firstHSListOrg[firstHSListOrg==0] = 0
            self.firstHSList = list(firstHSListOrg)
            totalErrorHS = list(np.ones_like(totalHSvalList))
        else:
            relativeErrorHS = []
            for firstErrorHSVar, errorHSVar in zip(self.firstHSList,totalHSvalList):
                if firstErrorHSVar == 0:
                    relativeErrorHS.append(errorHSVar/(self.scTol/len(totalHSvalList)))
                else:
                    relativeErrorHS.append(errorHSVar/firstErrorHSVar)  
            totalErrorHS = relativeErrorHS

        self.adaptiveHB.harmonicSetList = curHarmset
        return totalErrorHS

    def appendHarmonicSetList(self,
                              harmonicSet: list):
        r"""
        This method appends the harmonic set to the list of harmonic sets.
        Also updates the maximum harmonic set if the new harmonic set is larger.
        
        Parameters
        ----------
        harmonicSet : list
            The harmonic set to be appended.
        """
        
        # check if maximum harmonic set, and save it.
        for idxVar, harmSet in enumerate(harmonicSet):
            if len(harmSet) > len(self.maxHarmonicSet[idxVar]):
                self.maxHarmonicSet[idxVar] = harmSet
        
        self.harmonicSetListList.append(copy.deepcopy(harmonicSet))

    