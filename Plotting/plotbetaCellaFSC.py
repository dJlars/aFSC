r"""
Author:         Lars de Jong
Date:           2025-09-27
Description:    Main file for plotting the results of the electrophysilogy of beta cells. 
#
This file controls the process of generating the plots 
of the electrophysilogy of beta cells.

Plots to generate:
1. phase portrait
2. difference position
3. position statistics
4. HB convergence
5. SC convergence
6. Total error convergence
7. Total error compare convergence
8. CP over H
9. complexity over error
"""
import dill
import os
import sys

from plottingaFSC import *

# Add parent directory of aFSC to sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
aFSC_dir = os.path.join(base_dir,"..")
aFSC_dir = os.path.abspath(aFSC_dir)
sys.path.insert(0, aFSC_dir)

# load data and then call the plots

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

nrSamples = int(1e5)
# Build absolute path relative to the script directory
dataStr = f"./data_betaCell/0_betaCellFSC_plotingData_s{nrSamples:.0e}_bF5e-01.pkl"
dataStr2 = f"./data_betaCell/0_betaCellFSC_plotingData_s{nrSamples:.0e}_bF5e-02.pkl"
with open(dataStr,"rb") as f:
    betaCellaFSCData = dill.load(f)

with open(dataStr2,"rb") as f:
    betaCellaFSCData2 = dill.load(f)

hbConvData, scConvData, sparsGridData,\
    errorConvergenceData, cpOverHarmonics, \
        complexitOverErrorData, modelStochasticsList, \
            plotSamples = betaCellaFSCData

definedHBList, errorHBList, maxCPHB, = hbConvData
definedCSList, errorSCList, errorSCestimatorList, myHSet_sc = scConvData
betaCellaFSCModel, accuracyStr = sparsGridData
predefTotalErrorList, totalErrorExpectedValueList, \
    totalErrorMCEstimatorList, totalErrorMC_SCEstimatorList, \
    totalErrorMC_HBEstimatorList = errorConvergenceData
preDef_usedHSetListUnion, preDef_nrFSCModels, preDef_usedComplexityList,\
    preDef_legendTotalErrorList, preDef_usedIndexSetUnion = cpOverHarmonics
errorModelsList, allErrorModelList, allErrorModelLegend = complexitOverErrorData
scMeanPos, scPosQuantiles, diffPosMin, diffPosMax = modelStochasticsList[-1]
plotSamplesPos, plotSamplesVel = plotSamples

hbConvData2, scConvData2, sparsGridData2,\
    errorConvergenceData2, cpOverHarmonics2, \
        complexitOverErrorData2, modelStochasticsList2, \
            plotSamples2 = betaCellaFSCData2

definedHBList2, errorHBList2, maxCPHB2, = hbConvData2
definedCSList2, errorSCList2, errorSCestimatorList2, myHSet_sc2 = scConvData2
betaCellaFSCModel2, accuracyStr2 = sparsGridData2
predefTotalErrorList2, totalErrorExpectedValueList2, \
    totalErrorMCEstimatorList2, totalErrorMC_SCEstimatorList2, \
    totalErrorMC_HBEstimatorList2 = errorConvergenceData2
preDef_usedHSetListUnion2, preDef_nrFSCModels2, preDef_usedComplexityList2,\
    preDef_legendTotalErrorList2, preDef_usedIndexSetUnion2 = cpOverHarmonics2
errorModelsList2, allErrorModelList2, allErrorModelLegend2 = complexitOverErrorData2
scMeanPos2, scPosQuantiles2, diffPosMin2, diffPosMax2 = modelStochasticsList2[-1]
plotSamplesPos2, plotSamplesVel2 = plotSamples2

nrTimePoints = errorModelsList[-1].adaptiveHB.nrEvalPts
tVec = np.linspace(0,(2*np.pi),nrTimePoints)

result_folder = os.path.join(script_dir,"fig_betaCell")

variableList = [r'V', r'n', r'c']

xticksList = [
    [0, np.pi/2 ,np.pi, 3/2*np.pi, np.pi*2],
    [r"$0$", r"$\pi/2$", r"$\pi$", r"$3/2 \pi$", r"$2\pi$"]
]

# ------------------------
# --- Plotting section ---
# ------------------------

dpiVal = 600

plotLimiCycle(scMeanPos2[0,:], scMeanPos2[1,:], 
              plotSamplesPos2[:,:,0], 
              plotSamplesPos2[:,:,1],
              r"$V$ in $mV$",
              r"$n$ in $[-]$",
              r"$\left(\mathbb{E}[V(t,\boldsymbol{\theta})],\mathbb{E}[n(t,\boldsymbol{\theta})]\right)$",
              figName="betaCell_LimitCycle_bF5e-02",
              figDir=result_folder,
              figSize=(7/2.54,5/2.54),
              dpiVal=dpiVal)

plotDiffMinMax(diffPosMin2[0,:], diffPosMax2[0,:],
               tVec,
               variableList[0],
               r"normalized time",
               xTicksList=xticksList,
               figName="betaCell_Pos_Diff_bF5e-02_V",
               figDir=result_folder,
               figSize=(7/2.54,5/2.54),
               dpiVal=dpiVal)

plotMeanQuantiles(scMeanPos[0,:], scPosQuantiles[:,:,0],
                  tVec,
                  variableList[0],
                  r"normalized time",
                  xTicksList=xticksList,
                  figName="betaCell_Pos_Mean_bF5e-02_V",
                  figDir=result_folder,
                  figSize=(7/2.54,5/2.54),
                  dpiVal=dpiVal)

plotDiffMinMax(diffPosMin2[1,:], diffPosMax2[1,:],
               tVec,
               variableList[1],
               r"normalized time",
               xTicksList=xticksList,
               figName="betaCell_Pos_Diff_bF5e-02_n",
               figDir=result_folder,
               figSize=(7/2.54,5/2.54),
               dpiVal=dpiVal)

plotMeanQuantiles(scMeanPos[1,:], scPosQuantiles[:,:,1],
                  tVec,
                  variableList[1],
                  r"normalized time",
                  xTicksList=xticksList,
                  figName="betaCell_Pos_Mean_bF5e-02_n",
                  figDir=result_folder,
                  figSize=(7/2.54,5/2.54),
                  dpiVal=dpiVal)

plotDiffMinMax(diffPosMin2[2,:], diffPosMax2[2,:],
               tVec,
               variableList[2],
               r"normalized time",
               xTicksList=xticksList,
               figName="betaCell_Pos_Diff_bF5e-02_c",
               figDir=result_folder,
               figSize=(7/2.54,5/2.54),
               dpiVal=dpiVal)

plotMeanQuantiles(scMeanPos[2,:], scPosQuantiles[:,:,2],
                  tVec,
                  variableList[2],
                  r"normalized time",
                  xTicksList=xticksList,
                  figName="betaCell_Pos_Mean_bF5e-02_c",
                  figDir=result_folder,
                  figSize=(7/2.54,5/2.54),
                  dpiVal=dpiVal)

plotConver(definedHBList2, errorHBList2, 
           [0,1,2],['V', 'n', 'c'],
           r"Number of Harmonics $\#H$",
           figName=f'betaCell_HBconv_bF5e-02_s{nrSamples:.0e}',
           figDir=result_folder,
           figSize=(7/2.54,5/2.54),
           dpiVal=dpiVal,
           xTicksList=definedHBList2[::3])

plotConver(definedCSList2, errorSCList2, 
           [0,1,2],['V', 'n', 'c'],
           r"Number of CPs $\#\Lambda$",
           figName=f'betaCell_SCconv_bF5e-02_s{nrSamples:.0e}',
           figDir=result_folder,
           figSize=(7/2.54,5/2.54),
           dpiVal=dpiVal,)

plotTotalError(predefTotalErrorList2,
               totalErrorExpectedValueList2, 
                [0,1,2],['V', 'n', 'c'],
               figName='betaCell_TotalErrorConv_bF5e-02',
               figDir=result_folder,
               figSize=(8/2.54,5/2.54),
               dpiVal=dpiVal)

plotTotalErrorCompare(predefTotalErrorList, 
                      totalErrorExpectedValueList,
                      totalErrorExpectedValueList2, 
                      [0,1,2],['V', 'n', 'c'], 
                      figName='betaCell_TotalErrorConv_Compare',
                      figDir=result_folder,
                      figSize=(12/2.54,5/2.54),
                      dpiVal=dpiVal)

plotCPoverHarmonics(preDef_usedHSetListUnion2,
                    preDef_nrFSCModels2,
                    preDef_usedComplexityList2,
                    preDef_legendTotalErrorList2,
                    preDef_usedIndexSetUnion2,
                    variableList,
                    figName = 'betaCell_CPoverHarmonics_bF5e-02',
                    xTicksStep=2,
                    vertical=True,
                    figDir=result_folder,
                    figSize=(12/2.54,5/2.54),
                    dpiVal=dpiVal)

plotComplexityOverError(len(allErrorModelList2), #nr Models
                        errorModelsList2,
                        allErrorModelList2,
                        allErrorModelLegend2,
                        figName = 'betaCell_ComplexityoverError_bF5e-02',
                        figDir=result_folder,
                        figSize=(15/2.54,8/2.54),
                        dpiVal=dpiVal)

print("This is the way!")