r"""
Author:         Lars de Jong
Date:           2025-09-29
Description:    Main file for electrophysiology of beta cells. 
#
This file controls the process of generating the surrogate and 
all data for the convergence studies.
"""
import os
import copy
import dill

import chaospy as cp
import numpy as np

from aFSC.models.fourierStochasticCollocation import fourierStochasticCollocation
from aFSC.utils.idx_admissibility import update_active_set

from generalUtils.mainHelperFunctions import *

from betaCell.cellAHB import betaCellAHB

from generalUtils.argparser import get_config_Cell
from generalUtils.logger import init_logger
from generalUtils.filemanager import save_script

# ---------------------
# --- Setup section ---
# ---------------------

predefTotalErrorList = [
                    1e-2, #1e-3, 1e-4, 1e-5,
                    ]

np.random.seed(42)
nrSamples = int(1e1)

saveResultsPath = "./"

result_folder = save_script(saveResultsPath,
                            os.path.realpath(__file__),
                            "betaCell","", 
                            max_daily_folders = 7, 
                            max_res_folders = 7)

# logger, should be activated in the case of errors. 
# Due to iterative processes the log file may be to large to use
logger = None # init_logger(result_folder, "logfile")

balFac = 0.5

# Get user input from: cmd line > config file > defaults
configHB = get_config_Cell("betaCell/betaCellHB.yaml", result_folder, logger)

atpDist = cp.Beta(5,5,1730,1870)

distList = [atpDist]

jPDF = cp.J(atpDist)

varStrList = ['V','n','Ca']

folderModelStr = 'data_betaCell/betaCell'

def calculateFSCmodel(saveFSCStr,
                      hSet,
                      aHBBool,
                      fscError,
                      distList,
                      configHB,
                      balFac,
                      logger,
                      maxCP = 1000):
    
    if os.path.exists(saveFSCStr):
        with open(saveFSCStr, "rb") as f:
            myBetaCellFSC = dill.load(f)

    else:
        myAHBBetaCell = betaCellAHB([hSet, hSet, hSet],
                                configHB,
                                amp_s1 = -5.6,
                                logger=logger,
                                solThreshold = 0.5,
                                nonAdaptive=aHBBool,
                                getNewInitialGuess=True,
                                usedSamples=0.5,
                                )

        initialPCSet = [[0]]
        initialActiveSet = update_active_set(copy.deepcopy(initialPCSet))

        myBetaCellFSC = fourierStochasticCollocation(distList,
                                                    myAHBBetaCell,
                                                    fscError = fscError,
                                                    maxCP = maxCP,
                                                    scHBBalFac = balFac,
                                                    )

        myBetaCellFSC.calcFSCSurrogate([configHB.atp],
                                    initialPCSet,
                                    initialActiveSet)
        
        with open(saveFSCStr, "wb") as f:
            dill.dump(myBetaCellFSC, f)
    
    return myBetaCellFSC

# ----------------------
# --- Model calculation ----
# ----------------------

# --- Create samples ---
sampleStr = folderModelStr[:-8] + f'samples_uq{len(distList)}_s{nrSamples:.0e}.pkl'
samplesOrg = createSamples(sampleStr, jPDF, nrSamples)
samples = samplesOrg[np.newaxis, :]

# --- Total error models ---
totalErrorModelList = []
hCellInitSet = {1,2,3,4,5}
for totalError in predefTotalErrorList:
    saveTotalErroraFSCStr = folderModelStr + f"FSC_uq{len(distList)}_eps{totalError:.0e}.pkl"
    
    totalErBetaCellFSC = calculateFSCmodel(saveTotalErroraFSCStr,
                                       hCellInitSet,
                                       False, # aHB adptivity
                                       totalError,
                                       distList,
                                       configHB,
                                       balFac,
                                       logger,)
    
    totalErrorModelList.append(totalErBetaCellFSC)

print(f"Total error aFSC models with {len(predefTotalErrorList)} different total errors done!")

# ---------------------
# --- Predefined surrogates ---
# ---------------------
maxCPmaxHList = [
                [3, 10], #[3, 20], [3, 30],
                # [6, 10], [6, 20], [6, 30],
                # [9, 10], [9, 20], [9, 30],
                # [12, 10], [12, 20], [12, 30],
]

legendPredefinedList = []
givenModelList = []
usedIndexSetFixed = []
usedHSetFixedList = [[], [], []]
for maxCPmaxH in maxCPmaxHList:
    maxCP = maxCPmaxH[0]
    maxH = maxCPmaxH[1]

    hSet = set(np.arange(1,maxH+1))
    for idxVar1 in range(3):
        if len(hSet) > len(usedHSetFixedList[idxVar1]):
            usedHSetFixedList[idxVar1] = hSet

    legendPredefinedList.append(r'$\#\Lambda$ '+ f"{maxCP}, "+ r'$\#H$ ' + f"{len(hSet)}")
    savePreDefFSCStr = folderModelStr + f"FSC_uq{len(distList)}_mCP{maxCP}_mH{maxH}.pkl"

    predefinedBetaCellFSC = calculateFSCmodel(savePreDefFSCStr,
                                          hSet,
                                          True, # aHB adptivity
                                          1e-12,
                                          distList,
                                          configHB,
                                          balFac,
                                          logger,
                                          maxCP=maxCP)

    for curIndexFixed in predefinedBetaCellFSC.indexSetList:
        if tuple(curIndexFixed) not in usedIndexSetFixed:
            usedIndexSetFixed.append(tuple(curIndexFixed))

    givenModelList.append(predefinedBetaCellFSC)

print(f"Predefined aFSC models with {len(legendPredefinedList)} different maxCP and maxH done!")

# ---------------------
# --- Complexity calculation ---   
# ---------------------

# only for predefined models
preDef_errorComplexityArray, preDef_legendTotalErrorList, \
    preDef_usedIndexSet, preDef_usedHSetList, \
        preDef_usedIndexSetUnion, preDef_usedHSetListUnion, \
        preDef_usedComplexityList, preDef_allErrorModelList, \
        preDef_estimatorErrorList, preDef_complexityList = \
        complexityCalculation(totalErrorModelList,
                            predefTotalErrorList,
                            3,
                            [],
                            [],
                            [[],[],[]])

errorComplexityArray, legendTotalErrorList, \
    usedIndexSet, usedHSetList, \
        usedIndexSetUnion, usedHSetListUnion, \
        usedComplexityList, allErrorModelList, \
        estimatorErrorList, complexityList = \
        complexityCalculation(totalErrorModelList,
                            predefTotalErrorList,
                            3,
                            givenModelList,
                            usedIndexSetFixed,
                            usedHSetFixedList)

print(f"Complexity calculation with {len(allErrorModelList)} aFSC models done!")

# ---------------------
# --- MC HB sampling ---
# ---------------------
# MC sampling only need to be calcualted once, therefore extracted from following calculations

mcharmonicArray = np.arange(1, 61)
saveAHB_MC_Str = folderModelStr[:-8] + f'MC_uq{len(distList)}_maxH{len(mcharmonicArray)}_s{nrSamples:.0e}.pkl'

if os.path.exists(saveAHB_MC_Str):
    with open(saveAHB_MC_Str, "rb") as f:
        data = dill.load(f)
        mcAHBBetaCell, mcFourierCoeff = data
else:
    mcAHBBetaCell = betaCellAHB([set(mcharmonicArray), set(mcharmonicArray), set(mcharmonicArray)],
                                configHB,
                                amp_s1 = -5.6,
                                logger=logger,
                                solThreshold = 0.5,
                                nonAdaptive=True,
                                getNewInitialGuess=True,
                                usedSamples=0.5,
                                )
    
    mcFourierCoeff = np.zeros(((len(mcharmonicArray)*2+1)*3, nrSamples))
    initGuessOrg = mcAHBBetaCell.getInitialGuess([configHB.atp])

    idxMiddle = samples.shape[1]//2
    boolVar = True
    initGuess = initGuessOrg
    for idxSample in range(nrSamples):
        if idxSample < idxMiddle:
            curIdx = idxMiddle + idxSample
        else:
            if boolVar:
                initGuess = initGuessOrg
                boolVar = False
            curIdx = samples.shape[1] - idxSample -1
        sample = samples[:,curIdx]
        
        mcAHBBetaCell.assignUQParameters([sample])
        fourierCoeff, harmonicSet, hbAmpRatio = mcAHBBetaCell.adaptiveHB(initGuess)
        mcFourierCoeff[:, curIdx] = fourierCoeff

        # print every 10 % of the samples
        if not (nrSamples//10) == 0:
            if idxSample % (nrSamples//10) == 0:
                print(f'\nProgress: {idxSample/nrSamples*100:.0f}%')

    with open(saveAHB_MC_Str, "wb") as f:
        dill.dump([mcAHBBetaCell,mcFourierCoeff], f)

print(f"MC aHB sampling with {len(mcharmonicArray)} harmonics and {nrSamples} samples done!")


# ---------------------
# --- MC aFSC sampling ---
# ---------------------

modelStochasticsList = []
totalErrorMC_HBEstimatorList = []
totalErrorMC_SCEstimatorList = []
totalErrorMCEstimatorList = []
totalErrorExpectedValueList = []
nrMCCoeffIdx = len(mcharmonicArray)*2+1
for totalError in predefTotalErrorList:
    stochasticStr = folderModelStr + f'_fsc_stochastics_s{nrSamples:.0e}_uq{len(distList)}_eps{totalError:.0e}.pkl'
    stochasticStrMCPos = folderModelStr + f'_fsc_stochastics_s{nrSamples:.0e}_uq{len(distList)}_eps{totalError:.0e}_mcPos.pkl'
    stochasticStrSCPos = folderModelStr + f'_fsc_stochastics_s{nrSamples:.0e}_uq{len(distList)}_eps{totalError:.0e}_scPos.pkl'
    stochasticStrMCVel = folderModelStr + f'_fsc_stochastics_s{nrSamples:.0e}_uq{len(distList)}_eps{totalError:.0e}_mcVel.pkl'
    stochasticStrSCVel = folderModelStr + f'_fsc_stochastics_s{nrSamples:.0e}_uq{len(distList)}_eps{totalError:.0e}_scVel.pkl'
    stochasticStrPlotSamples = folderModelStr + f'_fsc_stochastics_s{nrSamples:.0e}_uq{len(distList)}_eps{totalError:.0e}_MCH{len(mcharmonicArray)}_PlotSampels.pkl'

    if os.path.exists(stochasticStr):

        with open(stochasticStr, "rb") as f:
            stochasticData = dill.load(f)

        mcMeanPos, mcMeanVel, scMeanPos, scMeanVel, \
            diffPosMin, diffPosMax, scPosQuantiles, scVelQuantiles, \
                mcPosQuantiles, mcVelQuantiles, aFSCTotalFourierCoeff, \
                    hbErrorEstimator, scErrorEstimator, \
                    errorEstimatorModel, normErrorsSC \
                      = stochasticData
        
        if totalError is predefTotalErrorList[-1]:
            with open(stochasticStrPlotSamples, "rb") as f:
                plotSamplesPos, plotSamplesVel = dill.load(f)
        
    else:
        specTotalStochStr = f'FSC_uq{len(distList)}_eps{totalError:.0e}'
        saveFSCStochStr = folderModelStr + specTotalStochStr 

        try:
            with open(saveFSCStochStr + '.pkl', "rb") as f:
                myStochBetaCellFSC = dill.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {saveFSCStochStr} not found. Please run the previous script to generate this file.")
    
        mcMeanPos, mcMeanVel, scMeanPos, scMeanVel, mcPos, mcVel, scPos, scVel, \
            diffPosMin, diffPosMax, scPosQuantiles, scVelQuantiles, \
                mcPosQuantiles, mcVelQuantiles, aFSCTotalFourierCoeff, \
                    plotSamplesPos, plotSamplesVel \
                        = calculateStochastics(myStochBetaCellFSC,
                                               mcAHBBetaCell,
                                               samples,
                                               mcFourierCoeff,
                                               False,
                                               int(0.1*nrSamples))

        hbErrorEstimator = sum(myStochBetaCellFSC.errorHBList) 
        scErrorEstimator = myStochBetaCellFSC.errorSCList[-1]
        errorEstimatorModel = hbErrorEstimator + scErrorEstimator
        
        with open(stochasticStrMCPos, "wb") as f:
            dill.dump(mcPos,f)
        with open(stochasticStrSCPos, "wb") as f:
            dill.dump(scPos,f)
        with open(stochasticStrMCVel, "wb") as f:
            dill.dump(mcVel,f)
        with open(stochasticStrSCVel, "wb") as f:
            dill.dump(scVel,f)

        normErrorsSC = calculateNorms(scPos,
                                      stochasticStrMCPos)
        
        with open(stochasticStr, "wb") as f:
            dill.dump([mcMeanPos, mcMeanVel, 
                    scMeanPos, scMeanVel,
                    diffPosMin, diffPosMax, 
                    scPosQuantiles, scVelQuantiles,
                    mcPosQuantiles, mcVelQuantiles,
                    aFSCTotalFourierCoeff,
                    hbErrorEstimator,
                    scErrorEstimator,
                    errorEstimatorModel,
                    normErrorsSC], f)

        del mcPos
        del scPos
        del mcVel
        del scVel
        
        if totalError is predefTotalErrorList[-1]:
            with open(stochasticStrPlotSamples, "wb") as f:
                dill.dump([plotSamplesPos, plotSamplesVel], f)
        
    totalErrorExpectedValueList.append(normErrorsSC)
        
    modelStochasticsList.append([scMeanPos, scPosQuantiles, \
                                diffPosMin, diffPosMax])
    totalErrorMC_HBEstimatorList.append(hbErrorEstimator)
    totalErrorMC_SCEstimatorList.append(scErrorEstimator)
    totalErrorMCEstimatorList.append(errorEstimatorModel)
    
    if totalError is predefTotalErrorList[-1]:
        plotSamples = [plotSamplesPos, plotSamplesVel]

print(f"MC aFSC sampling with {nrSamples} samples for {len(predefTotalErrorList)} aFSC models done!")

# ---------------------
# --- SC convergence ---
# ---------------------

definedCSList = [
                2, #4, 6, 8, 10,
                ]

definedSCModelList = []
errorSCestimatorList = []
errorSCExactList = []
myHSet_sc = set(np.arange(1, 41))
for maxCPSC in definedCSList:
    specSCStr = f'FSC_uq{len(distList)}_defCP{maxCPSC}_defNrH{len(myHSet_sc)}'
    saveFSC_SC_Str = folderModelStr + specSCStr + '.pkl'

    sc_BetaCellaFSC = calculateFSCmodel(saveFSC_SC_Str,
                                    myHSet_sc,
                                    True, # aHB adptivity
                                    1e-8,
                                    distList,
                                    configHB,
                                    balFac,
                                    logger,
                                    maxCP=maxCPSC)
    
    definedSCModelList.append(sc_BetaCellaFSC)

    if os.path.exists(saveFSC_SC_Str[:-4] + f'_s{nrSamples:.0e}_CoeffMat.pkl'):
        with open(saveFSC_SC_Str[:-4] + f'_s{nrSamples:.0e}_CoeffMat.pkl', "rb") as f:
            aFSCSCFourierCoeff,normErrorsSC = dill.load(f)
    else:
        nrCoeffs = sum([len(sc_BetaCellaFSC.maxHarmonicSet[i])*2 + 1 for i in range(3)])
        aFSCSCFourierCoeff = np.zeros((nrCoeffs, nrSamples))
        
        idxMiddle = samples.shape[1]//2
        scPos = np.zeros((nrSamples,sc_BetaCellaFSC.adaptiveHB.nrEvalPts,sc_BetaCellaFSC.adaptiveHB.nrVar))
        for idxSHB in range(nrSamples):

            if idxSHB < idxMiddle:
                curIdx1 = idxMiddle + idxSHB
            else:
                curIdx1 = samples.shape[1] - idxSHB -1
            sample = samples[:,curIdx1]
            aFSCSC = sc_BetaCellaFSC.stochasticCollocationEval([sample])
            aFSCSCFourierCoeff[:, curIdx1] = aFSCSC

            sc_fourierCoeffsList, omegaSC = sc_BetaCellaFSC.adaptiveHB.splitCoeffVec(aFSCSC)

            for idxVar in range(sc_BetaCellaFSC.adaptiveHB.nrVar):
                sc_pos, sc_vel, sc_acc = sc_BetaCellaFSC.adaptiveHB.calculatePositionVelocityAcceleration(sc_fourierCoeffsList[idxVar], 
                                                                omegaSC, 
                                                                idxVar)
                
                scPos[curIdx1,:,idxVar] = sc_pos

        normErrorsSC = calculateNorms(scPos,
                                      stochasticStrMCPos)
        del scPos
        
        with open(saveFSC_SC_Str[:-4] + f'_s{nrSamples:.0e}_CoeffMat.pkl', "wb") as f:
            dill.dump([aFSCSCFourierCoeff,normErrorsSC], f)
    
    errorSCestimatorList.append(sc_BetaCellaFSC.errorSCList[-1])
    errorSCExactList.append(normErrorsSC)

print(f"SC convergence with {nrSamples} samples for {len(definedCSList)} predefined models done!")


# ---------------------
# --- HB convergence ---
# ---------------------

maxCPHB = 1
definedHBList = [
                5, #10, 15, 20, 25, 30, 35, 40, 45, 50,
                ]

mcAHBBetaCell = betaCellAHB([set(mcharmonicArray), set(mcharmonicArray), set(mcharmonicArray)],
                                configHB,
                                amp_s1 = -5.6,
                                logger=logger,
                                solThreshold = 0.5,
                                nonAdaptive=True,
                                getNewInitialGuess=True,
                                usedSamples=0.5,
                                )

errorHBList = []
for curH in definedHBList:
    myHSet = set(np.arange(1, curH+1))
    saveFSC_HB_Str = folderModelStr + f'_HB_uq{len(distList)}_defH{curH}.pkl'
    hb_BetaCellaFSC = calculateFSCmodel(saveFSC_HB_Str,
                                    myHSet,
                                    True, # aHB adptivity
                                    1e-8,
                                    distList,
                                    configHB,
                                    balFac,
                                    logger,
                                    maxCP=1)

    meanSample = np.array(cp.E(atpDist),)[np.newaxis,np.newaxis]
    mcAHBBetaCell.assignUQParameters(meanSample)
    initGuess = mcAHBBetaCell.getInitialGuess([configHB.atp])
    fourierCoeff, harmonicSet, hbAmpRatio = mcAHBBetaCell.adaptiveHB(initGuess)
    mean_fourierCoeffsList, mean_omega = mcAHBBetaCell.splitCoeffVec(fourierCoeff)

    aFSCHB = hb_BetaCellaFSC.stochasticCollocationEval(meanSample)
    hb_fourierCoeffsList, omegaHB = hb_BetaCellaFSC.adaptiveHB.splitCoeffVec(aFSCHB)

    hbPos = np.zeros((1,hb_BetaCellaFSC.adaptiveHB.nrEvalPts,hb_BetaCellaFSC.adaptiveHB.nrVar))
    meanPos = np.zeros((1,mcAHBBetaCell.nrEvalPts,mcAHBBetaCell.nrVar))
    for idxVarHB in range(hb_BetaCellaFSC.adaptiveHB.nrVar):
        hb_pos, _, _ = hb_BetaCellaFSC.adaptiveHB.calculatePositionVelocityAcceleration(hb_fourierCoeffsList[idxVarHB], 
                                                        omegaHB, 
                                                        idxVarHB)
        
        mean_pos, _, _ = mcAHBBetaCell.calculatePositionVelocityAcceleration(mean_fourierCoeffsList[idxVarHB], 
                                                        mean_omega, 
                                                        idxVarHB)
        
        meanPos[0,:,idxVarHB] = mean_pos
        hbPos[0,:,idxVarHB] = hb_pos

    normErrorsHB = calculateNorms(hbPos,meanPos)

    errorHBList.append(normErrorsHB)
print(f"HB convergence with {maxCPHB} CPs for {len(definedHBList)} predefined models done!")

# ---------------------
# --- Save data for plotting ---
# ---------------------

# store data in appropiate lists for correspinding grafics

hbConvergenceData = [definedHBList, errorHBList, maxCPHB]
scConvergenceData = [definedCSList, errorSCExactList, errorSCestimatorList, myHSet_sc]

# model for sparse grid data
# idxModel = predefTotalErrorList.index(1e-5)
# sparseGridData = [totalErrorModelList[idxModel],"1e-5"]
idxModel = predefTotalErrorList.index(1e-2)
sparseGridData = [totalErrorModelList[idxModel],"1e-2"]

errorConvergenceData = [predefTotalErrorList,totalErrorExpectedValueList,
                        totalErrorMCEstimatorList,
                        totalErrorMC_SCEstimatorList,
                        totalErrorMC_HBEstimatorList]

cpOverHarmonics = [preDef_usedHSetListUnion,
                   len(predefTotalErrorList),
                   preDef_usedComplexityList,
                   preDef_legendTotalErrorList,
                   preDef_usedIndexSetUnion]

complexitOverErrorData = [totalErrorModelList, 
                          allErrorModelList,
                          legendTotalErrorList + legendPredefinedList]

aFSCbetaCellData = [hbConvergenceData,
                    scConvergenceData,
                    sparseGridData,
                    errorConvergenceData,
                    cpOverHarmonics,
                    complexitOverErrorData,
                    modelStochasticsList,
                    plotSamples]

saveStrPlotData = folderModelStr[:-7] + "0_betaCellFSC_plotingData.pkl"

with open(saveStrPlotData,"wb") as f:
    dill.dump(aFSCbetaCellData,f)

print("This is the way.")
