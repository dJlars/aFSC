r"""
Author:         Lars de Jong
Date:           2025-10-01
Description:    Main file for plotting the results of the Duffing oscillator with cubic term. 
#
This file controls the process of generating the plots 
of the Duffing oscillator with cubic term.

Plots to generate:
1. phase portrait
2. difference position
3. position statistics
4. HB convergence
5. SC convergence
6. Total error convergence
7. CP over H
8. complexity over error
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
dataStr = f"./data_duffing_x3/0_duffingx3FSC_plotingData_s{nrSamples:.0e}.pkl"
with open(dataStr,"rb") as f:
    duffingaFSCData = dill.load(f)

hbConvData, scConvData, sparsGridData,\
    errorConvergenceData, cpOverHarmonics, \
        complexitOverErrorData, modelStochasticsList, \
            plotSamples  = duffingaFSCData

definedHBList, errorHBList, maxCPHB, = hbConvData
definedCSList, errorSCList, errorSCestimatorList, myHSet_sc = scConvData
duffingaFSCModel, accuracyStr = sparsGridData
predefTotalErrorList, totalErrorExpectedValueList, \
    totalErrorMCEstimatorList = errorConvergenceData
usedHSetListUnion, nrFSCModels, usedComplexityList,\
    legendTotalErrorList, usedIndexSetUnion = cpOverHarmonics
errorModelsList, allErrorModelList, allErrorModelLegend = complexitOverErrorData
plotSamplesPos, plotSamplesVel = plotSamples

scMeanPos, scMeanVel, scPosQuantiles, diffPosMin, diffPosMax = modelStochasticsList[-1]
omega = errorModelsList[-1].adaptiveHB.omega
nrTimePoints = errorModelsList[-1].adaptiveHB.nrEvalPts
tVec = np.linspace(0,(2*np.pi/omega),nrTimePoints)

result_folder = os.path.join(script_dir,"fig_duffing_x3")

variableList = ['x']

# ------------------------
# --- Plotting section ---
# ------------------------

dpiVal = 600
plotLimiCycle(scMeanPos, scMeanVel, 
              plotSamplesPos, 
              plotSamplesVel,
              r"$x$",
              r"$\dot{x}$",
              r"$\left(\mathbb{E}[x(t,\boldsymbol{\theta})],\mathbb{E}[v(t,\boldsymbol{\theta})]\right)$",
              figName="duffingx3_LimitCycle",
              figDir=result_folder,
              figSize=(8/2.54,5/2.54),
              dpiVal=dpiVal)

plotDiffMinMax(diffPosMin, diffPosMax,
               tVec,
               variableList[0],
               r"$t$",
               figName="duffingx3_Pos_Diff",
               figDir=result_folder,
               figSize=(8/2.54,5/2.54),
               dpiVal=dpiVal)

plotMeanQuantiles(scMeanPos, scPosQuantiles.squeeze(),
                  tVec,
                  variableList[0],
                  r"$t$",
                  figName="duffingx3_Pos_Mean",
                  figDir=result_folder,
                  figSize=(8/2.54,5/2.54),
                  dpiVal=dpiVal)

plotConver(definedHBList, errorHBList, 
           [0],['x'], 
           r"Number of Harmonics $\#H$",
           figName=f'duffingx3_HBconv_s{nrSamples:.0e}',
           figDir=result_folder,
           figSize=(12/2.54,5/2.54),
           dpiVal=dpiVal)

plotConver(definedCSList, errorSCList, 
           [0],['x'], 
           r"Number of CPs $\#\Lambda$",
           figName=f'duffingx3_SCconv_s{nrSamples:.0e}',
           figDir=result_folder,
           figSize=(12/2.54,5/2.54),
           dpiVal=dpiVal)

plotTotalError(predefTotalErrorList, 
               totalErrorExpectedValueList,
               [0],['x'],
               figName='duffingx3_TotalErrorConv',
               figDir=result_folder,
               figSize=(12/2.54,5/2.54),
               dpiVal=dpiVal)

plotCPoverHarmonics(usedHSetListUnion,
                    nrFSCModels,
                    usedComplexityList,
                    legendTotalErrorList,
                    usedIndexSetUnion,
                    variableList,
                    figName = 'duffingx3_CPoverHarmonics',
                    vertical=True,
                    figDir=result_folder,
                    figSize=(14.5/2.54,5/2.54),
                    dpiVal=dpiVal)

plotComplexityOverError(len(allErrorModelList), #nr Models
                        errorModelsList,
                        allErrorModelList,
                        allErrorModelLegend,
                        figName = 'duffingx3_ComplexityoverError',
                        figDir=result_folder,
                        figSize=(15/2.54,8/2.54),
                        dpiVal=dpiVal)


print("This is the way!")