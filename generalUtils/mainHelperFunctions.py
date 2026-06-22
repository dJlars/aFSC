r"""
Author:         Lars de Jong
Date:           2025-06-15
Description:    Helper functions for the main files. 
#
This module contains utility functions that are used for each 
example main file. These functions are designed to
simplify the main files by providing common functionalities
such as reading computing aFSC models and postprocessing data for
visualization.
"""
import os
import sys
import dill
import copy

import numpy as np
import chaospy as cp

from aFSC.models.fourierStochasticCollocation import fourierStochasticCollocation
from aFSC.models.adaptiveHB import adaptiveHarmonicBalance

def createSamples(saveStr: str,
                  jPDF: cp.J,
                  nrSamples: int):
    r"""
    Function to create samples from a joint probability distribution function (jPDF)

    Parameters
    ----------
    saveStr : str
        Path to save the samples
    jPDF : cp.J
        Joint probability distribution function from chaospy
    nrSamples : int
        Number of samples to generate
    
    Returns
    ----------
    samples : np.ndarray
        Sorted array of samples generated from the jPDF
    """
    if os.path.exists(saveStr):
        with open(saveStr, "rb") as f:
            samples = dill.load(f)
    else:
        samples = np.sort(jPDF.sample(nrSamples))

        directory = os.path.dirname(saveStr)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(saveStr, "wb") as f:
            dill.dump(samples, f)
    
    print((f"{nrSamples} samples created!"))

    return samples

def calculateComplexity(totalErrorModelList: list,
                        totalErrorList: list,
                        d: int):
    r"""
    Function to calculate the complexity of the total error models.
    In addition, calculates the used CPs and harmonic sets for each variable in the total error models.
    
    Parameters
    ----------
    totalErrorModelList : list
        List of total error models
    totalErrorList : list
        List of total errors for each model
    d : int
        Number of variables in the model

    Returns
    ----------
    errorComplexityArray : np.ndarray
        Array containing the complexity of the total error models
    legendTotalErrorList : list
        List of legends for the total errors
    usedIndexSet : list
        List of used index sets for the total error models
    usedHSetList : list
        List of used harmonic sets for each variable in the total error models
    """

    legendTotalErrorList = []
    errorComplexityList = []
    usedIndexSet = []
    usedHSetList = [set([]) for _ in range(d)]
    for totalErroraFSC, totalError in zip(totalErrorModelList, totalErrorList):
        nrCP = len(totalErroraFSC.collocationPointList) 
        nrHList = []
        for harmonicSetsCP in totalErroraFSC.harmonicSetListList:
            nrHVar = [len(setVar) for setVar in harmonicSetsCP]
            nrHList.append(sum(nrHVar))
        nrH = max(nrHList)
        errorComplexityList.append([totalError, nrCP, nrH, np.sum(np.array(nrHList))])

        idxAdm = -len(totalErroraFSC.addedAdmissibleSet)
        if idxAdm == 0:
            idxAdm = len(totalErroraFSC.indexSetList)
        # extract cps and hs for plotting
        for curIndex in totalErroraFSC.indexSetList[:idxAdm]:
            if tuple(curIndex) not in usedIndexSet:
                usedIndexSet.append(tuple(curIndex))
        for idxVar, maxHVarSet in enumerate(totalErroraFSC.maxHarmonicSet):
            usedHSetList[idxVar].update(maxHVarSet)

        legendTotalErrorList.append(r"$\boldsymbol{\delta}_{\text{FSC}}$ "+ f"{totalError:.0e}, "+ r'$\#\Lambda$ '+ f"{nrCP}, "+ r'$\#H$ ' + f"{nrH}")

    errorComplexityArray = np.array(errorComplexityList)

    return errorComplexityArray, legendTotalErrorList, usedIndexSet, usedHSetList

def calc_H_CP_Union(usedIndexSet: list,
                    usedIndexSetFixed: list,
                    usedHSetList: list,
                    usedHSetFixedList: list,
                    errorModelsList: list,
                    givenModelList: list,):
    r"""
    Function to calculate the union of used index sets and harmonic sets for the total error models.

    Parameters
    ----------
    usedIndexSet : list
        List of used index sets for the total error models
    usedIndexSetFixed : list
        List of used index sets for the predefined models
    usedHSetList : list
        List of used harmonic sets for each variable in the total error models
    usedHSetFixedList : list
        List of used harmonic sets for each variable in the predefined models
    errorModelsList : list
        List of total error models
    givenModelList : list
        List of predefined models

    Returns
    ----------
    usedIndexSetUnion : list
        List of union of used index sets for the total error models and predefined models
    usedHSetListUnion : list
        List of union of used harmonic sets for each variable in the total error models and predefined models
    usedComplexityList : list
        List of used complexity matrices for each model
    allErrorModelList : list
        List of all error models, including total error models and predefined models
    """

    usedIndexSetUnion = copy.deepcopy(usedIndexSet)
    for curIndexFixed in usedIndexSetFixed:
        if tuple(curIndexFixed) not in usedIndexSet:
            usedIndexSetUnion.append(tuple(curIndexFixed))
    usedHSetListUnion = [set(uHFVar).union(set(uHVar)) for uHFVar, uHVar in zip(usedHSetFixedList, usedHSetList)]
    allErrorModelList = errorModelsList + givenModelList

    # iterate through usedCP and usedH for each model and save in 2d array
    usedComplexityList = []
    for idxModel, currentModel in enumerate(allErrorModelList):
        complexityMatList = []
        if currentModel.adaptiveHB.sameH4all:
            nrVar = 1
        else:
            nrVar = len(currentModel.maxHarmonicSet)
        for idxVar in range(nrVar):
            complexityMatVar = np.zeros((len(usedIndexSetUnion), len(usedHSetListUnion[idxVar])), dtype=int)
            for idxCP, curIndex in enumerate(usedIndexSetUnion):
                for idxH, curH in enumerate(usedHSetListUnion[idxVar]):

                    if hasattr(currentModel,'addedAdmissibleSet'):
                        admInd = -len(currentModel.addedAdmissibleSet)
                        if admInd == 0:
                            admInd = len(currentModel.indexSetList)
                    else:
                        admInd = len(currentModel.indexSetList)
                    # hier dann nicht nach max checken sondern wie viele hs an dem punkt
                    if idxModel <= len(errorModelsList):
                        # if list(curIndex) in currentModel.indexSetList[:-len(currentModel.errorSCAdmissibleList[idxVar][-1])+1]:
                        if list(curIndex) in currentModel.indexSetList[:admInd]:
                            idxCPDet = currentModel.indexSetList.index(list(curIndex))
                            if curH in currentModel.harmonicSetListList[idxCPDet][idxVar]:
                                complexityMatVar[idxCP, idxH] = 1
                    else:
                        if list(curIndex) in currentModel.indexSetList[:admInd]:
                            idxCPDet = currentModel.indexSetList.index(list(curIndex))
                            if curH in currentModel.harmonicSetListList[idxCPDet][idxVar]:
                                complexityMatVar[idxCP, idxH] = 1

            complexityMatList.append(complexityMatVar)
        
        usedComplexityList.append(complexityMatList)

    return usedIndexSetUnion, usedHSetListUnion, \
        usedComplexityList, allErrorModelList


def calculateErrorComplexity(allErrorModelList: list,
                            ):
    r"""
    Function to calculate the error from the estimator as well as the complexity of the model.

    Parameters
    ----------
    allErrorModelList : list
        List of all error models, including total error models and predefined models
    
    Returns
    ----------
    errorEstimatorList : list
        List of estimator errors for each model
    complexityList : list
        List of complexity for each model, calculated as the sum of the lengths of the harmonic 
        sets for all CPS
    """

    complexityList = []
    errorEstimatorList = []
    for model in allErrorModelList:
        actualError = model.errorHBList[-1] + model.errorSCList[-1]
        totalComplexity = 0
        for harmonicSetList in model.harmonicSetListList:#[:-len(model.addedAdmissibleSet)]:
            for harmonicSet in harmonicSetList:
                totalComplexity += len(harmonicSet)

        errorEstimatorList.append(actualError)
        complexityList.append(totalComplexity)

    return errorEstimatorList, complexityList

def complexityCalculation(totalErrorModelList: list,
                           totalErrorList: list,
                           d: float,
                           givenModelList: list,
                           usedIndexSetFixed: list,
                           usedHSetFixedList: list):
    r"""
    Function to calculate the complexity of the total error models and the predefined models.
    
    Parameters
    ----------
    totalErrorModelList : list
        List of total error models
    totalErrorList : list
        List of total errors for each model
    d : float
        Number of variables in the model
    givenModelList : list
        List of predefined models
    usedIndexSetFixed : list
        List of used index sets for the predefined models
    usedHSetFixedList : list
        List of used harmonic sets for each variable in the predefined models

    Returns
    ----------
    errorComplexityArray : np.ndarray
        Array containing the complexity of the total error models
    legendTotalErrorList : list
        List of legends for the total error models
    usedIndexSet : list
        List of used index sets for the total error models
    usedHSetList : list
        List of used harmonic sets for each variable in the total error models
    usedIndexSetUnion : list
        List of union of used index sets for the total error models and predefined models
    usedHSetListUnion : list
        List of union of used harmonic sets for each variable in the total error models and predefined models
    usedComplexityList : list
        List of used complexity matrices for each model
    allErrorModelList : list
        List of all error models, including total error models and predefined models
    estimatorErrorList : list
        List of estimator errors for each model
    complexityList : list
        List of complexity for each model, calculated as the sum of the lengths of the harmonic 
        sets for all CPS
    """

    errorComplexityArray, legendTotalErrorList, \
        usedIndexSet, usedHSetList = calculateComplexity(totalErrorModelList,
                                                        totalErrorList,
                                                        d)

    usedIndexSetUnion, usedHSetListUnion, \
            usedComplexityList, allErrorModelList = \
                calc_H_CP_Union(usedIndexSet,
                                usedIndexSetFixed,
                                usedHSetList,
                                usedHSetFixedList,
                                totalErrorModelList,
                                givenModelList)


    estimatorErrorList, complexityList = calculateErrorComplexity(allErrorModelList)

    return errorComplexityArray, legendTotalErrorList, \
        usedIndexSet, usedHSetList, \
        usedIndexSetUnion, usedHSetListUnion, \
        usedComplexityList, allErrorModelList, \
        estimatorErrorList, complexityList

def calculateStochastics(myaFSCModel: fourierStochasticCollocation,
                         mcAHBModel: adaptiveHarmonicBalance,
                         samples: np.ndarray,
                         mcFourierCoeff: np.ndarray,
                         velOption: bool = False,
                         nrPlotSamples: int = int(1e1)):
    r"""
    Function to calculate the stochastics of the MC and aFSC model.

    Parameters
    ----------
    myaFSCModel : fourierStochasticCollocation
        aFSC model to calculate the stochastics for
    mcAHBModel : adaptiveHarmonicBalance
        MC model to calculate the stochastics for
    samples : np.ndarray
        Samples to evaluate the aFSC model
    mcFourierCoeff : np.ndarray
        Fourier coefficients of the MC model

    Returns
    ----------
    mcMeanPos : np.ndarray
        Mean position of the MC model
    mcMeanVel : np.ndarray
        Mean velocity of the MC model
    scMeanPos : np.ndarray
        Mean position of the aFSC model
    scMeanVel : np.ndarray
        Mean velocity of the aFSC model
    mcPos : np.ndarray 
        Position of the MC model for each sample
    mcVel : np.ndarray
        Velocity of the MC model for each sample
    scPos : np.ndarray
        Position of the aFSC model for each sample
    scVel : np.ndarray
        Velocity of the aFSC model for each sample
    diffPosMin : np.ndarray
        Minimum difference between the MC and aFSC model positions
    diffPosMax : np.ndarray
        Maximum difference between the MC and aFSC model positions
    scPosQuantiles : np.ndarray
        Quantiles of the position of the aFSC model
    scVelQuantiles : np.ndarray
        Quantiles of the velocity of the aFSC model
    mcPosQuantiles : np.ndarray
        Quantiles of the position of the MC model
    mcVelQuantiles : np.ndarray
        Quantiles of the velocity of the MC model
    aFSCTotalFourierCoeff : np.ndarray
        Total Fourier coefficients of the aFSC model
    """
    
    nrCoeffs = sum([len(myaFSCModel.maxHarmonicSet[i])*2 + 1 for i in range(myaFSCModel.adaptiveHB.nrVar)])
    aFSCTotalFourierCoeff = np.zeros((nrCoeffs, samples.shape[1]))

    mcPos = np.zeros((samples.shape[1],mcAHBModel.nrEvalPts,mcAHBModel.nrVar))
    if velOption:
        mcVel = np.zeros((samples.shape[1],mcAHBModel.nrEvalPts,mcAHBModel.nrVar))
    else:
        mcVel = []
    mcOmega = np.zeros((samples.shape[1]))

    scPos = np.zeros((samples.shape[1],mcAHBModel.nrEvalPts,mcAHBModel.nrVar))
    if velOption:
        scVel = np.zeros((samples.shape[1],mcAHBModel.nrEvalPts,mcAHBModel.nrVar))
    else:
        scVel = []
    scOmega = np.zeros((samples.shape[1]))

    idxMiddle = samples.shape[1]//2
    myaFSCModel.adaptiveHB.harmonicSetList = myaFSCModel.maxHarmonicSet
    for idxSample in range(samples.shape[1]):
        if idxSample < idxMiddle:
            curIdx = idxMiddle + idxSample
        else:
            curIdx = samples.shape[1] - idxSample -1
        sample = samples[:,curIdx]

        mc_fourierCoeffsList, omegaMC = mcAHBModel.splitCoeffVec(mcFourierCoeff[:, curIdx])

        if not (isinstance(sample,list) or isinstance(sample,np.ndarray)):
            sample = [sample]

        sc_fourierCoeffs = myaFSCModel.stochasticCollocationEval(sample)
        aFSCTotalFourierCoeff[:, curIdx] = sc_fourierCoeffs
        sc_fourierCoeffsList, omegaSC = myaFSCModel.adaptiveHB.splitCoeffVec(sc_fourierCoeffs)

        for idxVar in range(mcAHBModel.nrVar):
            mcCoeff = mc_fourierCoeffsList[idxVar]
            mc_pos, mc_vel, mc_acc = \
                mcAHBModel.calculatePositionVelocityAcceleration(mcCoeff,
                                                            omegaMC, 
                                                            idxVar)
            mcPos[curIdx,:,idxVar] = mc_pos
            if velOption:
                mcVel[curIdx,:,idxVar] = mc_vel

            scCoeff = sc_fourierCoeffsList[idxVar]
            sc_pos, sc_vel, sc_acc = \
                myaFSCModel.adaptiveHB.calculatePositionVelocityAcceleration(scCoeff, 
                                                            omegaSC, 
                                                            idxVar)
            
            scPos[curIdx,:,idxVar] = sc_pos
            if velOption:
                scVel[curIdx,:,idxVar] = sc_vel

        mcOmega[curIdx] = omegaMC
        scOmega[curIdx] = omegaSC
        # print every 10 % of the samples
        if not (samples.shape[1]//10) == 0:
            if idxSample % (samples.shape[1]//10) == 0:
                print(f'Progress: {idxSample/samples.shape[1]*100:.0f}%')

    mcMeanPos = np.mean(mcPos, axis=0).T
    if velOption:
        mcMeanVel = np.mean(mcVel, axis=0).T
    else:
        mcMeanVel = []

    scMeanPos = np.mean(scPos, axis=0).T
    if velOption:
        scMeanVel = np.mean(scVel, axis=0).T
    else:
        scMeanVel = []

    diffPosMin_org = np.min(np.abs(mcPos-scPos), axis=0).T
    diffPosMax = np.max(np.abs(mcPos-scPos), axis=0).T
    diffPosMin = diffPosMin_org
    diffPosMin[np.abs(diffPosMin) <= sys.float_info.epsilon] = sys.float_info.epsilon     

    scPosQuantiles = np.quantile(scPos, [0.025, 0.975], axis=0)
    scVelQuantiles = []#np.quantile(scVel, [0.025, 0.975], axis=0)

    mcPosQuantiles = np.quantile(mcPos, [0.025, 0.975], axis=0)
    mcVelQuantiles = []#np.quantile(mcVel, [0.025, 0.975], axis=0)

    # Extract random samples for later plotting
    rng = np.random.default_rng(42)
    cols2Plot = rng.choice(scPos.shape[0], size=nrPlotSamples, replace=False)
    plotSamplesPos = scPos[cols2Plot,:,:]
    if velOption:
        plotSamplesVel = scVel[cols2Plot,:,:]
    else:
        plotSamplesVel = []

    return mcMeanPos, mcMeanVel, scMeanPos, scMeanVel, mcPos, mcVel, scPos, scVel, \
            diffPosMin, diffPosMax, scPosQuantiles, scVelQuantiles, \
                mcPosQuantiles, mcVelQuantiles, aFSCTotalFourierCoeff, \
                plotSamplesPos, plotSamplesVel


def calculateNorms(scPos: np.ndarray,
                   stochasticStrMCPos: str):
    """
    Function to calculate the various norms for the aFSC model.

    Parameters
    ----------
    scPos : np.ndarray
        position of the aFSC model
    stochasticStrMCPos : string
        string with path and file of stored position array of the MC study

    Returns
    ----------
    normErrorList : list
        List with the four error norms: 
        - relative root mean square error
        - relative maximum root square error
        - relative maximum error
        - relative maximum root mean square error
    
    """
    
    if isinstance(stochasticStrMCPos, str):
        with open(stochasticStrMCPos, 'rb') as f:
            mcPosReload = dill.load(f)
    else:
        mcPosReload = stochasticStrMCPos

    # axis 0: samples
    # axis 1: time steps
    # axis 3: variables
    
    quadIntTime_num = 1/scPos.shape[1]*np.sum(np.sum((mcPosReload-scPos)**2,axis=2), axis=1)
    quadIntTime_denom = 1/scPos.shape[1]*np.sum(np.sum(scPos**2,axis=2), axis=1)

    quadIntTime2_num = 1/scPos.shape[1]*np.sum((mcPosReload-scPos)**2, axis=1)
    quadIntTime2_denom = 1/scPos.shape[1]*np.sum(scPos**2, axis=1)

    rootMeanSquareErrVar_rel = np.sqrt(1/quadIntTime2_num.shape[0]* np.sum(quadIntTime2_num/quadIntTime2_denom, axis=0))

    maxRootSquareErrorVar = np.max(np.sqrt(quadIntTime2_num), axis = 0)
    maxRootSquareErrorVar_rel = maxRootSquareErrorVar/np.max(np.sqrt(quadIntTime2_denom), axis = 0)

    diff = np.abs(mcPosReload-scPos)
    maximumNorm = np.max(np.max(diff, axis=1), axis=0)

    rmse_theta_per_t = np.sqrt(np.mean(diff**2, axis=0))   # over samples
    err_L2theta_LinfT = np.max(rmse_theta_per_t, axis=0)

    diff_rel = np.abs(mcPosReload)
    maximumNorm_rel = maximumNorm/np.sqrt(np.mean(quadIntTime2_denom, axis=0))

    rmse_theta_per_t_rel = np.sqrt(np.mean(diff_rel**2, axis=0))   # over samples
    err_L2theta_LinfT_rel = err_L2theta_LinfT/np.max(rmse_theta_per_t_rel, axis=0)

    normErrorList = [rootMeanSquareErrVar_rel,
                     maxRootSquareErrorVar_rel,
                     maximumNorm_rel, 
                     err_L2theta_LinfT_rel]
    del mcPosReload, scPos

    return normErrorList
