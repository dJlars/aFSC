r"""
Author:         Lars de Jong
Date:           2025-09-29
Description:    Main file for the Duffing oscillator with sine term. 
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

from duffing.duffingHB import duffingHB

from generalUtils.argparser import get_config_Duffingx3
from generalUtils.logger import init_logger
from generalUtils.filemanager import save_script

# ---------------------
# --- Setup section ---
# ---------------------

predefTotalErrorList = [
                    1e-1, #1e-2, 1e-3, 1e-4, 1e-5,
                    ]

np.random.seed(42)
nrSamples = int(1e1)

saveResultsPath = "./"

result_folder = save_script(saveResultsPath,
                            os.path.realpath(__file__),
                            "duffing","", 
                            max_daily_folders = 7, 
                            max_res_folders = 7)

# logger, should be activated in the case of errors. 
# Due to iterative processes the log file may be to large to use
logger = None #init_logger(result_folder, "logfile")

# Get user input from: cmd line > config file > defaults
configHB = get_config_Duffingx3("duffing/duffingHB.yaml", result_folder, logger)

alphaDist = cp.Beta(5,5,6,8)
betaDist = cp.Beta(5,5,4,6)
deltaDist = cp.Uniform(0.01,0.02)
gammaDist = cp.Uniform(0.2,0.5)

distList = [alphaDist, betaDist, 
            deltaDist, gammaDist]

jPDF = cp.J(alphaDist, betaDist,
            deltaDist, gammaDist)

folderModelStr = 'data_duffing/duffing'

def calculateFSCmodel(saveFSCStr,
                      hSet,
                      aHBBool,
                      fscError,
                      distList,
                      configHB,
                      logger,
                      maxCP = 1000):
    
    if os.path.exists(saveFSCStr):
        with open(saveFSCStr, "rb") as f:
            myDuffingFSC = dill.load(f)

    else:
        # determined in seperate run with fixed harmonics
        deflationFakeSolution = [[np.array([-2.08869161e-18,  6.67218610e-02,  6.67837935e-04,  2.33666412e-18, -6.11353719e-19, -4.12213407e-06, -9.90704061e-08,  4.18638670e-19, -5.68996071e-19,  4.51439146e-10,  1.84686433e-11, -2.89095200e-19, -3.76003515e-19, -6.10012246e-14, -3.50635184e-15,  1.38766965e-19, -5.76092328e-19, 1.21369086e-17,  8.40583863e-19]),
                                np.array([-3.38170702e-17,  2.30672856e+00,  9.49311878e-01,  2.07455394e-17, 2.32549314e-17, -4.21345513e-02, -9.85388887e-02, -1.13252501e-18, -5.80277929e-18, -4.07358125e-03,  1.03508626e-02,  6.08303672e-18, -2.93482882e-18,  1.25878328e-03, -5.58229467e-04,  7.37469845e-19,  7.06607855e-18, -1.74154138e-04, -6.57601122e-05])],
                                [[{1, 2, 3, 4, 5, 6, 7, 8, 9}],
                                 [{1, 2, 3, 4, 5, 6, 7, 8, 9}]]]

        myAHBDuffing = duffingHB([hSet],
                                configHB,
                                logger=logger,
                                # solThreshold = 0.5,
                                deflationFakeSolutions = deflationFakeSolution,
                                RelRangeRndGuess = [0,2],
                                nonAdaptive = aHBBool,
                                deflation = True,
                                maxDefSol=1,
                                usedSamples=0.5,
                                )

        initialPCSet = [[0,0,0,0]]
        initialActiveSet = update_active_set(copy.deepcopy(initialPCSet))

        myDuffingFSC = fourierStochasticCollocation(distList,
                                                    myAHBDuffing,
                                                    fscError = fscError,
                                                    maxCP = maxCP,
                                                    )

        myDuffingFSC.calcFSCSurrogate(myAHBDuffing.getCP(),
                                    initialPCSet,
                                    initialActiveSet)
        
        with open(saveFSCStr, "wb") as f:
            dill.dump(myDuffingFSC, f)
    
    return myDuffingFSC

# ----------------------
# --- Model calculation ----
# ----------------------

# --- Create samples ---
sampleStr = folderModelStr + f'_samples_uq{len(distList)}_s{nrSamples:.0e}.pkl'
samples = createSamples(sampleStr, jPDF, nrSamples)

# --- Total error models ---
totalErrorModelList = []
hCellInitSet = {1}
for totalError in predefTotalErrorList:
    saveTotalErroraFSCStr = folderModelStr + f"FSC_uq{len(distList)}_eps{totalError:.0e}.pkl"
    
    totalErBetaCellFSC = calculateFSCmodel(saveTotalErroraFSCStr,
                                       hCellInitSet,
                                       False, # aHB adptivity
                                       totalError,
                                       distList,
                                       configHB,
                                       logger,)
    
    totalErrorModelList.append(totalErBetaCellFSC)

print(f"Total error aFSC models with {len(predefTotalErrorList)} different total errors done!")

# ---------------------
# --- Predefined surrogates ---
# ---------------------

maxCPmaxHList = [
                 [5, 3], #[5, 5],
                #  [9, 5], [9, 7],
                #  [15, 7], [15, 9],
                #  [25, 9], [25,11],
                #  [40,11], [40,15],
                #  [60,15], [60,17],
]

legendPredefinedList = []
givenModelList = []
usedIndexSetFixed = []
usedHSetFixedList = [[]]
for maxCPmaxH in maxCPmaxHList:
    maxCP = maxCPmaxH[0]
    maxH = maxCPmaxH[1]

    hSet = set(np.arange(1,maxH+1,2))
    if len(hSet) > len(usedHSetFixedList[-1]):
        usedHSetFixedList[-1] = hSet

    legendPredefinedList.append(r'$\#\Lambda$ '+ f"{maxCP}, "+ r'$\#H$ ' + f"{len(hSet)}")
    savePreDefFSCStr = folderModelStr + f"FSC_uq{len(distList)}_mCP{maxCP}_mH{maxH}.pkl"

    predefinedDuffingFSC = calculateFSCmodel(savePreDefFSCStr,
                                            hSet,
                                            True, # aHB adptivity
                                            1e-12,
                                            distList,
                                            configHB,
                                            logger,
                                            maxCP=maxCP)

    for curIndexFixed in predefinedDuffingFSC.indexSetList:#[:-len(predefinedDuffingFSC.addedAdmissibleSet)]:
        if tuple(curIndexFixed) not in usedIndexSetFixed:
            usedIndexSetFixed.append(tuple(curIndexFixed))

    givenModelList.append(predefinedDuffingFSC)

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
                            1, # nr variables
                            [],
                            [],
                            [[]])

errorComplexityArray, legendTotalErrorList, \
    usedIndexSet, usedHSetList, \
        usedIndexSetUnion, usedHSetListUnion, \
        usedComplexityList, allErrorModelList, \
        estimatorErrorList, complexityList = \
        complexityCalculation(totalErrorModelList,
                            predefTotalErrorList,
                            1, # nr variables
                            givenModelList,
                            usedIndexSetFixed,
                            usedHSetFixedList)

print(f"Complexity calculation with {len(allErrorModelList)} aFSC models done!")

# ---------------------
# --- MC HB sampling ---
# ---------------------
# MC sampling only need to be calcualted once, therefore extracted from following calculations

mcharmonicArray = np.arange(1, 26,2)
saveAHB_MC_Str = folderModelStr + f'MC_uq{len(distList)}_maxH{len(mcharmonicArray)}_s{nrSamples:.0e}.pkl'

if os.path.exists(saveAHB_MC_Str):
    with open(saveAHB_MC_Str, "rb") as f:
        data = dill.load(f)
        mcAHBDuffing, mcFourierCoeff = data
else:

    # determined in seperate run with fixed harmonics
    deflationFakeSolution = [[np.array([-2.08869161e-18,  6.67218610e-02,  6.67837935e-04,  2.33666412e-18, -6.11353719e-19, -4.12213407e-06, -9.90704061e-08,  4.18638670e-19, -5.68996071e-19,  4.51439146e-10,  1.84686433e-11, -2.89095200e-19, -3.76003515e-19, -6.10012246e-14, -3.50635184e-15,  1.38766965e-19, -5.76092328e-19, 1.21369086e-17,  8.40583863e-19]),
                            np.array([-3.38170702e-17,  2.30672856e+00,  9.49311878e-01,  2.07455394e-17, 2.32549314e-17, -4.21345513e-02, -9.85388887e-02, -1.13252501e-18, -5.80277929e-18, -4.07358125e-03,  1.03508626e-02,  6.08303672e-18, -2.93482882e-18,  1.25878328e-03, -5.58229467e-04,  7.37469845e-19,  7.06607855e-18, -1.74154138e-04, -6.57601122e-05])],
                            [[{1, 2, 3, 4, 5, 6, 7, 8, 9}],
                                [{1, 2, 3, 4, 5, 6, 7, 8, 9}]]]
    
    mcAHBDuffing = duffingHB([set(mcharmonicArray)],
                                configHB,
                                logger=logger,
                                # solThreshold = 0.5,
                                deflationFakeSolutions = deflationFakeSolution,
                                RelRangeRndGuess = [0,2],
                                nonAdaptive = True,
                                deflation = True,
                                maxDefSol=1,
                                usedSamples=0.5,
                                )
    
    mcFourierCoeff = np.zeros(((len(mcharmonicArray)*2+1), int(nrSamples)))
    initGuessOrg = mcAHBDuffing.getInitialGuess(0)

    idxMiddle = len(samples)//2
    boolVar = True
    initGuess = initGuessOrg
    for idxSample in range(int(nrSamples)):
        if idxSample < idxMiddle:
            curIdx = idxMiddle + idxSample
        else:
            if boolVar:
                initGuess = initGuessOrg
                boolVar = False
            curIdx = len(samples) - idxSample -1
        sample = samples[:,curIdx]
        
        mcAHBDuffing.assignUQParameters(sample)
        fourierCoeff, harmonicSet, hbAmpRatio = mcAHBDuffing.adaptiveHB(initGuess)
        mcFourierCoeff[:, curIdx] = fourierCoeff

        # print every 10 % of the samples
        if not (samples.shape[1]//10) == 0:
            if idxSample % (samples.shape[1]//10) == 0:
                print(f'\nProgress: {idxSample/samples.shape[1]*100:.0f}%')

    with open(saveAHB_MC_Str, "wb") as f:
        dill.dump([mcAHBDuffing,mcFourierCoeff], f)

print(f"MC aHB sampling with {len(mcharmonicArray)} harmonics and {nrSamples} samples done!")

# ---------------------
# --- MC aFSC sampling ---
# ---------------------

modelStochasticsList = []
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
                myStochDuffingFSC = dill.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {saveFSCStochStr} not found. Please run the previous script to generate this file.")
    
        mcMeanPos, mcMeanVel, scMeanPos, scMeanVel, mcPos, mcVel, scPos, scVel, \
            diffPosMin, diffPosMax, scPosQuantiles, scVelQuantiles, \
                mcPosQuantiles, mcVelQuantiles, aFSCTotalFourierCoeff, \
                     plotSamplesPos, plotSamplesVel \
                        = calculateStochastics(myStochDuffingFSC,
                                               mcAHBDuffing,
                                               samples,
                                               mcFourierCoeff,
                                               True,
                                               int(0.1*nrSamples))

        errorEstimatorModel = sum(myStochDuffingFSC.errorHBList) + myStochDuffingFSC.errorSCList[-1]

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
        
    modelStochasticsList.append([scMeanPos, scMeanVel, scPosQuantiles, \
                                diffPosMin, diffPosMax])
    totalErrorMCEstimatorList.append(errorEstimatorModel)

    if totalError is predefTotalErrorList[-1]:
        plotSamples = [plotSamplesPos, plotSamplesVel]

print(f"MC aFSC sampling with {nrSamples} samples for {len(predefTotalErrorList)} aFSC models done!")

# ---------------------
# --- SC convergence ---
# ---------------------

definedCSList = [
                10, #20, 30, 40, 50, 60, 70, 80, 90, 100,
                ]

definedSCModelList = []
errorSCList = []
errorSCestimatorList = []
errorSCExactList = []
myHSet_sc = set(np.arange(1, 12, 2))
for maxCPSC in definedCSList:
    specSCStr = f'FSC_uq{len(distList)}_defCP{maxCPSC}_defNrH{len(myHSet_sc)}'
    saveFSC_SC_Str = folderModelStr + specSCStr + '.pkl'

    sc_DuffingFSC = calculateFSCmodel(saveFSC_SC_Str,
                                    myHSet_sc,
                                    True, # aHB adptivity
                                    1e-8,
                                    distList,
                                    configHB,
                                    logger,
                                    maxCP=maxCPSC)
    
    definedSCModelList.append(sc_DuffingFSC)

    if os.path.exists(saveFSC_SC_Str[:-4] + f'_s{nrSamples:.0e}_CoeffMat.pkl'):
        with open(saveFSC_SC_Str[:-4] + f'_s{nrSamples:.0e}_CoeffMat.pkl', "rb") as f:
            aFSCSCFourierCoeff,normErrorsSC = dill.load(f)
    else:
        nrCoeffs = sum([len(sc_DuffingFSC.maxHarmonicSet[i])*2 + 1 for i in range(sc_DuffingFSC.adaptiveHB.nrVar)])
        aFSCSCFourierCoeff = np.zeros((nrCoeffs, nrSamples))
        
        idxMiddle = samples.shape[1]//2
        scPos = np.zeros((nrSamples,sc_DuffingFSC.adaptiveHB.nrEvalPts,sc_DuffingFSC.adaptiveHB.nrVar))
        for idxSHB in range(nrSamples):

            if idxSHB < idxMiddle:
                curIdx1 = idxMiddle + idxSHB
            else:
                curIdx1 = samples.shape[1] - idxSHB -1
            sample = samples[:,curIdx1]
            aFSCSC = sc_DuffingFSC.stochasticCollocationEval(sample)
            aFSCSCFourierCoeff[:, curIdx1] = aFSCSC

            sc_fourierCoeffsList, omegaSC = sc_DuffingFSC.adaptiveHB.splitCoeffVec(aFSCSC)

            for idxVar in range(sc_DuffingFSC.adaptiveHB.nrVar):
                sc_pos, sc_vel, sc_acc = sc_DuffingFSC.adaptiveHB.calculatePositionVelocityAcceleration(sc_fourierCoeffsList[idxVar], 
                                                                omegaSC, 
                                                                idxVar)
                
                scPos[curIdx1,:,idxVar] = sc_pos
        
        normErrorsSC = calculateNorms(scPos,
                                      stochasticStrMCPos)
        del scPos

        with open(saveFSC_SC_Str[:-4] + f'_s{nrSamples:.0e}_CoeffMat.pkl', "wb") as f:
            dill.dump([aFSCSCFourierCoeff,normErrorsSC], f)
    
    errorSCestimatorList.append(sc_DuffingFSC.errorSCList[-1])
    errorSCExactList.append(normErrorsSC)

print(f"SC convergence with {nrSamples} samples for {len(definedCSList)} predefined models done!")

# ---------------------
# --- HB convergence ---
# ---------------------

maxCPHB = 1
definedHBList = [
                1, #3, 5, 7, 9, 11
                ]


errorHBList = []
for curH in definedHBList:
    myHSet = set(np.arange(1, curH+1))
    saveFSC_HB_Str = folderModelStr + f'_HB_uq{len(distList)}_defH{curH}.pkl'
    hb_DuffingaFSC = calculateFSCmodel(saveFSC_HB_Str,
                                    myHSet,
                                    True, # aHB adptivity
                                    1e-8,
                                    distList,
                                    configHB,
                                    logger,
                                    maxCP=1)
    
    meanSample = np.array([cp.E(alphaDist),cp.E(betaDist),
                           cp.E(deltaDist),cp.E(gammaDist)])
    mcAHBDuffing.assignUQParameters(meanSample)
    initGuess = mcAHBDuffing.getInitialGuess(0)
    fourierCoeff, harmonicSet, hbAmpRatio = mcAHBDuffing.adaptiveHB(initGuess)
    mean_fourierCoeffsList, mean_omega = mcAHBDuffing.splitCoeffVec(fourierCoeff)

    aFSCHB = hb_DuffingaFSC.stochasticCollocationEval(meanSample)
    hb_fourierCoeffsList, omegaHB = hb_DuffingaFSC.adaptiveHB.splitCoeffVec(aFSCHB)

    hbPos = np.zeros((1,hb_DuffingaFSC.adaptiveHB.nrEvalPts,hb_DuffingaFSC.adaptiveHB.nrVar))
    meanPos = np.zeros((1,mcAHBDuffing.nrEvalPts,mcAHBDuffing.nrVar))
    for idxVarHB in range(hb_DuffingaFSC.adaptiveHB.nrVar):
        hb_pos, _, _ = hb_DuffingaFSC.adaptiveHB.calculatePositionVelocityAcceleration(hb_fourierCoeffsList[idxVarHB], 
                                                        omegaHB, 
                                                        idxVarHB)
        
        mean_pos, _, _ = mcAHBDuffing.calculatePositionVelocityAcceleration(mean_fourierCoeffsList[idxVarHB], 
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
idxModel = predefTotalErrorList.index(1e-1)
sparseGridData = [totalErrorModelList[idxModel],"1e-1"]

errorConvergenceData = [predefTotalErrorList,totalErrorExpectedValueList,totalErrorMCEstimatorList]

cpOverHarmonics = [preDef_usedHSetListUnion,
                   len(predefTotalErrorList),
                   preDef_usedComplexityList,
                   preDef_legendTotalErrorList,
                   preDef_usedIndexSetUnion]

complexitOverErrorData = [totalErrorModelList, 
                          allErrorModelList,
                          legendTotalErrorList + legendPredefinedList]

aFSCduffingData = [hbConvergenceData,
                   scConvergenceData,
                   sparseGridData,
                   errorConvergenceData,
                   cpOverHarmonics,
                   complexitOverErrorData,
                   modelStochasticsList,
                   plotSamples]

saveStrPlotData = folderModelStr[:-7] + "0_duffingFSC_plotingData.pkl"

with open(saveStrPlotData,"wb") as f:
    dill.dump(aFSCduffingData,f)

print("This is the way!")