r"""
Author:         Lars de Jong
Date:           2025-09-29
Description:    Contains all plotting routines for the aFSC method. 
#
This file contains all methods to generate all figures 
for the aFSC method.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import tab20
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm

def saveFigures(fig: plt.figure, 
                figName: str, 
                figDir: str,
                dpiVal: float):
    r"""
    Save the figure to the specified directory with the given name.
    
    Parameters
    ----------
    fig : matplotlib.figure
        The figure to save.
    figName : str
        The name of the figure file (without extension).
    figDir : str
        The directory where the figure will be saved.
    dpiVal : float
        Resolution of pdf figure in dpi
    """
    if not os.path.exists(figDir):
        os.makedirs(figDir)
    
    figPath = os.path.join(figDir, f"{figName}.pdf")
    fig.savefig(figPath, bbox_inches='tight', dpi=dpiVal)
    print(f"{figName} Figure saved to {figPath} with resolution {dpiVal}")

def plotConver(definedDomainList, 
               normErrorDomainList,
               varIdxList,
               varStrList,
               xLabelStr,
               figName,
               figDir=None,
               figSize=(5,5),
               dpiVal=300,
               xTicksList=None):
    r"""
    Convergence plots for frequency and uncertain domain.
    
    Parameters
    ----------
    definedDomainList : list
        List of frequency or uncertain domain points
    normErrorDomainList : list
        List of error norms at each provided domain point
    varIdxList : list
        List of indices for which the convergence study is performed
    varStrList : list
        List of str for each variabl the convergence study is performed
    xLabelStr : str
        String of the evaluated domain for the x axis label
    figName : str
        The name of the figure file (without extension).
    figDir : str
        The directory where the figure will be saved.
    figSize : tuple
        Figure Size as tuple.
    dpiVal : float
        Resolution of pdf figure in dpi
    xTicksList : list
        List of x-axis ticks 
    """
    
    plt.style.use("./Plotting/dissertation.mplstyle")
    
    for modelVarOIIdx, modelVrOIStr in zip(varIdxList, varStrList):
        globalMeanSquareErrorVar_relList = []
        meanSquareErrVar_relList = []
        maxSquareErrorVar_relList = []
        maximumNorm_relList = []
        err_L2theta_LinfT_relList = []
        for i in range(len(definedDomainList)):
            globalMeanSquareErrorVar_relList.append(normErrorDomainList[i][1])
            meanSquareErrVar_relList.append(normErrorDomainList[i][3][modelVarOIIdx])
            maxSquareErrorVar_relList.append(normErrorDomainList[i][5][modelVarOIIdx])
            maximumNorm_relList.append(normErrorDomainList[i][7][modelVarOIIdx])
            err_L2theta_LinfT_relList.append(normErrorDomainList[i][9][modelVarOIIdx])

        normErrorListExtracted = [meanSquareErrVar_relList,
                                  maxSquareErrorVar_relList,
                                  maximumNorm_relList,
                                  err_L2theta_LinfT_relList]
        
        normErrorStringList = [r'$\varepsilon_{\mathbb{E}L_T^2,d}$',
                               r'$\varepsilon_{L_{\boldsymbol{\theta}}^{\infty}L_T^2,d}$',
                               r'$\varepsilon_{L_{\boldsymbol{\theta}}^{\infty} L_T^{\infty},d}$',
                               r'$\varepsilon_{L_T^{\infty}RMS_{\boldsymbol{\theta}},d}$']
        
        fig2, ax2 = plt.subplots(figsize=figSize)
        ax2.set_xlabel(xLabelStr)
        ax2.set_ylabel(r'$\varepsilon$')

        for normError, normStr in zip(normErrorListExtracted,normErrorStringList):
            ax2.plot(definedDomainList,normError,'*-', label=normStr, markersize = 4)

        ax2.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0,
        )
        if xTicksList is None:
            xTicksList = definedDomainList
        ax2.set_xticks(xTicksList)
        ax2.set_xticklabels([str(int(val)) for val in xTicksList])
        ax2.set_yscale('log')
        ax2.grid(True)

        plt.tight_layout()

        if figDir is not None:
            saveFigures(fig2, figName+'Var_'+modelVrOIStr, figDir, dpiVal=dpiVal)

def plot4DSpareGrid(myaFSCModel,
                    figName,
                    figDir=None,
                    figSize=(5,5),
                    dpiVal=300):
    r"""
    Sparse grid plot of a aFSC model with 4 uncertain domains.
    
    Parameters
    ----------
    myaFSCModel : fourierStochasticCollocation
        Object of the aFSC model from which the 4D sparse grid will be plotted
    figName : str
        The name of the figure file (without extension).
    figDir : str
        The directory where the figure will be saved.
    figSize : tuple
        Figure Size as tuple.
    dpiVal : float
        Resolution of pdf figure in dpi
    """
    
    plt.style.use("./Plotting/dissertation.mplstyle")
    
    zDataList = np.array([len(curHSet[0]) for curHSet in myaFSCModel.harmonicSetListList])

    alphaCPs = myaFSCModel.uncertainDimList[0].knots
    betaCPs = myaFSCModel.uncertainDimList[1].knots
    deltaCPs = myaFSCModel.uncertainDimList[2].knots
    gammaCPs = myaFSCModel.uncertainDimList[3].knots
    uqCPList = [alphaCPs, betaCPs, deltaCPs, gammaCPs]

    indexList = myaFSCModel.indexSetList

    varLabelList = [r'$\alpha$', r'$\beta$', r'$\delta$', r'$\gamma$']
    
    # Plot with discrete colormap
    cmap = plt.get_cmap('plasma', len(np.unique(zDataList)))
    colors = cmap(np.linspace(0, 1, len(np.unique(zDataList))))
    discrete_viridis = ListedColormap(np.vstack(([1, 1, 1, 1], colors)))

    bounds = [0] + list(np.unique(zDataList)) + [np.unique(zDataList)[-1]+1]
    norm = BoundaryNorm(bounds, discrete_viridis.N)

    fig3, ax3 = plt.subplots(2,3,figsize=figSize)
    fig3.subplots_adjust(wspace=0.02, hspace=0.15)
    
    max_width = max(len(uqCPList[i]) for i in range(4))
    max_height = max(len(uqCPList[i]) for i in range(4))

    idxPlotH = 0
    idxPlotV = 0
    ims = []
    for firstIdx in range(3, -1, -1):
        for secondIdx in range(firstIdx - 1, -1, -1):
            
            selectedzDataList = []
            selectedIndexList = []
            tupleList = []
            for idxNr, indexSet in enumerate(indexList):
                curtuple = tuple([indexSet[firstIdx], indexSet[secondIdx]])
                if curtuple not in tupleList:
                    tupleList.append(curtuple)
                    selectedzDataList.append(zDataList[idxNr])
                    selectedIndexList.append(indexSet)

            z_grid = np.zeros((len(uqCPList[secondIdx]), len(uqCPList[firstIdx])))

            for index, zVal in zip(selectedIndexList, selectedzDataList):
                z_grid[index[secondIdx], index[firstIdx]] = int(zVal)

            im = ax3[idxPlotV, idxPlotH].imshow(z_grid, cmap=discrete_viridis, 
                                            norm=norm, origin='lower')
            ims.append(im)
            
            ax3[idxPlotV, idxPlotH].set_xticks(np.arange(max_width))
            ax3[idxPlotV, idxPlotH].set_yticks(np.arange(max_height))
            ax3[idxPlotV, idxPlotH].set_xticklabels([f"{val:.0f}" for val in range(max_width)])
            ax3[idxPlotV, idxPlotH].set_yticklabels([f"{val:.0f}" for val in range(max_height)])
            ax3[idxPlotV, idxPlotH].set_xlabel(varLabelList[firstIdx])
            ax3[idxPlotV, idxPlotH].set_ylabel(varLabelList[secondIdx])        
            
            if not idxPlotH == 0:
                ax3[idxPlotV, idxPlotH].yaxis.labelpad = 0.5

            ax = ax3[idxPlotV, idxPlotH]
            ax.set_axisbelow(True)
            grid_x = np.arange(0.5, max_width, 1)
            grid_y = np.arange(0.5, max_height, 1)
            ax.set_xticks(grid_x, minor=True)
            ax.set_yticks(grid_y, minor=True)
            ax.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.8, alpha=0.5)
            ax.tick_params(which='minor', size=0)
            # Advance subplot indices
            if idxPlotH < 2:
                idxPlotH += 1
            else:
                if idxPlotV < 1:
                    idxPlotV += 1
                    idxPlotH = 0

    for i in range(2):
        for j in range(3):
            ax = ax3[i, j]
            ax.set_xlim(-0.5, max_width - 0.5)
            ax.set_ylim(-0.5, max_height - 0.5)

    for i in range(2):
        for j in range(3):
            ax = ax3[i, j]
            
            if i != 1:
                ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
                ax.xaxis.set_label_position('top')
            else:
                ax.tick_params(axis='x', top=False, bottom=True, labeltop=False, labelbottom=True)
                ax.xaxis.set_label_position('bottom')
            
            if j != 0:
                ax.set_yticks([])
                ax.set_yticklabels([])
    
    z_unique = np.unique(zDataList)  
    bounds_array = np.array(bounds)

    tick_positions = []
    tick_labels = []

    # Always include 0 (white background center)
    interval_idx = np.where((bounds_array[:-1] <= 0) & (0 < bounds_array[1:]))[0]
    if len(interval_idx) > 0:
        lower, upper = bounds_array[interval_idx[0]], bounds_array[interval_idx[0] + 1]
        tick_positions.append((lower + upper) / 2)  # e.g., 1.0 or 0.5
        tick_labels.append('0')

    # Add centers for actual data values
    for z_val in sorted(z_unique):
        interval_idx = np.where((bounds_array[:-1] <= z_val) & (z_val < bounds_array[1:]))[0]
        if len(interval_idx) > 0:
            lower, upper = bounds_array[interval_idx[0]], bounds_array[interval_idx[0] + 1]
            tick_positions.append((lower + upper) / 2)
            tick_labels.append(str(int(z_val)))

    cbar = fig3.colorbar(ims[0], ax=ax3, ticks=tick_positions)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label('Number of Harmonics', rotation=90, labelpad=5)

    # plt.tight_layout()

    if figDir is not None:
        saveFigures(fig3, figName, figDir, dpiVal=dpiVal)


def plot2DSpareGrid(myaFSCModel,
                    varLabelList,
                    figName,
                    figDir=None,
                    figSize=(5,5),
                    dpiVal=300):
    r"""
    Sparse grid plot of a aFSC model with 2 uncertain domains.
    
    Parameters
    ----------
    myaFSCModel : fourierStochasticCollocation
        Object of the aFSC model from which the 4D sparse grid will be plotted
    varLabelList : list
        List of the uncertain variables 
    figName : str
        The name of the figure file (without extension).
    figDir : str
        The directory where the figure will be saved.
    figSize : tuple
        Figure Size as tuple.
    dpiVal : float
        Resolution of pdf figure in dpi
    """
    
    plt.style.use("./Plotting/dissertation.mplstyle")
    
    zDataList = np.array([len(curHSet[0]) for curHSet in myaFSCModel.harmonicSetListList])

    firstCPs = myaFSCModel.uncertainDimList[0].knots
    secondCPs = myaFSCModel.uncertainDimList[1].knots
    uqCPList = [firstCPs, secondCPs]

    indexList = myaFSCModel.indexSetList

    # Plot with discrete colormap
    cmap = plt.get_cmap('plasma', len(np.unique(zDataList)))
    colors = cmap(np.linspace(0, 1, len(np.unique(zDataList))))
    discrete_viridis = ListedColormap(np.vstack(([1, 1, 1, 1], colors)))

    bounds = [0] + list(np.unique(zDataList)) + [np.unique(zDataList)[-1]+1]
    norm = BoundaryNorm(bounds, discrete_viridis.N)

    fig3, ax3 = plt.subplots(figsize=figSize)
    
    # Compute global max dimensions for uniform limits
    max_width = max(len(uqCPList[i]) for i in range(2))   # max over all possible firstIdx
    max_height = max(len(uqCPList[i]) for i in range(2))  # max over all possible secondIdx

    idxPlotH = 0
    idxPlotV = 0
    ims = []  
    for firstIdx in range(1, -1, -1):
        for secondIdx in range(firstIdx - 1, -1, -1):
            
            selectedzDataList = []
            selectedIndexList = []
            tupleList = []
            for idxNr, indexSet in enumerate(indexList):
                curtuple = tuple([indexSet[firstIdx], indexSet[secondIdx]])
                if curtuple not in tupleList:
                    tupleList.append(curtuple)
                    selectedzDataList.append(zDataList[idxNr])
                    selectedIndexList.append(indexSet)

            z_grid = np.zeros((len(uqCPList[secondIdx]), len(uqCPList[firstIdx])))

            for index, zVal in zip(selectedIndexList, selectedzDataList):
                z_grid[index[secondIdx], index[firstIdx]] = int(zVal)

            im = ax3.imshow(z_grid, cmap=discrete_viridis, 
                                            norm=norm, origin='lower')
            ims.append(im)
        
            ax3.set_xticks(np.arange(max_width))
            ax3.set_yticks(np.arange(max_height))
            ax3.set_xticklabels([f"{val:.0f}" for val in range(max_width)])
            ax3.set_yticklabels([f"{val:.0f}" for val in range(max_height)])
            ax3.set_xlabel(varLabelList[firstIdx])
            ax3.set_ylabel(varLabelList[secondIdx])        
            
            if not idxPlotH == 0:
                ax3.yaxis.labelpad = 0.5

            ax3.set_axisbelow(True)
            grid_x = np.arange(0.5, max_width, 1)  
            grid_y = np.arange(0.5, max_height, 1)
            ax3.set_xticks(grid_x, minor=True)
            ax3.set_yticks(grid_y, minor=True)
            ax3.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.8, alpha=0.5)
            ax3.tick_params(which='minor', size=0)
            # Advance subplot indices
            if idxPlotH < 2:
                idxPlotH += 1
            else:
                if idxPlotV < 1:
                    idxPlotV += 1
                    idxPlotH = 0

    # Uniform limits for alignment
    ax3.set_xlim(-0.5, max_width - 0.5)
    ax3.set_ylim(-0.5, max_height - 0.5)

    z_unique = np.unique(zDataList)
    bounds_array = np.array(bounds)

    tick_positions = []
    tick_labels = []

    # Always include 0 (white background center)
    interval_idx = np.where((bounds_array[:-1] <= 0) & (0 < bounds_array[1:]))[0]
    if len(interval_idx) > 0:
        lower, upper = bounds_array[interval_idx[0]], bounds_array[interval_idx[0] + 1]
        tick_positions.append((lower + upper) / 2)
        tick_labels.append('0')

    # Add centers for actual data values
    for z_val in sorted(z_unique):
        interval_idx = np.where((bounds_array[:-1] <= z_val) & (z_val < bounds_array[1:]))[0]
        if len(interval_idx) > 0:
            lower, upper = bounds_array[interval_idx[0]], bounds_array[interval_idx[0] + 1]
            tick_positions.append((lower + upper) / 2)
            tick_labels.append(str(int(z_val)))

    cbar = fig3.colorbar(ims[0], ax=ax3, ticks=tick_positions)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label('Number of Harmonics', rotation=90, labelpad=5)

    # plt.tight_layout()

    if figDir is not None:
        saveFigures(fig3, figName, figDir, dpiVal=dpiVal)


def plotTotalError(predefTotalErrorList,
                   normErrorSCList,
                   varIdxList,
                   varStrList,
                   figName,
                   figDir=None,
                   figSize=(5,5),
                   dpiVal=300):
    r"""
    Convergence plot of severel aFSC model with different threshold.
    
    Parameters
    ----------
    predefTotalErrorList : list
        List of the predefined threshold of the aFSC models
    normErrorSCList : list
        List of error norms for each provided aFSC model
    varIdxList : list
        List of indices for which the convergence study is performed
    varStrList : list
        List of str for each variabl the convergence study is performed
    figName : str
        The name of the figure file (without extension).
    figDir : str
        The directory where the figure will be saved.
    figSize : tuple
        Figure Size as tuple.
    dpiVal : float
        Resolution of pdf figure in dpi
    """
    
    plt.style.use("./Plotting/dissertation.mplstyle")
    
    for modelVarOIIdx, modelVrOIStr in zip(varIdxList, varStrList):
        globalMeanSquareErrorVar_relList = []
        meanSquareErrVar_relList = []
        maxSquareErrorVar_relList = []
        maximumNorm_relList = []
        err_L2theta_LinfT_relList = []
        for i in range(len(predefTotalErrorList)):
            globalMeanSquareErrorVar_relList.append(normErrorSCList[i][1])
            meanSquareErrVar_relList.append(normErrorSCList[i][3][modelVarOIIdx])
            maxSquareErrorVar_relList.append(normErrorSCList[i][5][modelVarOIIdx])
            maximumNorm_relList.append(normErrorSCList[i][7][modelVarOIIdx])
            err_L2theta_LinfT_relList.append(normErrorSCList[i][9][modelVarOIIdx])

        normErrorListExtracted = [meanSquareErrVar_relList,
                                  maxSquareErrorVar_relList,
                                  maximumNorm_relList,
                                  err_L2theta_LinfT_relList]
        
        normErrorStringList = [r'$\varepsilon_{\mathbb{E}L_T^2,d}$',
                               r'$\varepsilon_{L_{\boldsymbol{\theta}}^{\infty}L_T^2,d}$',
                               r'$\varepsilon_{L_{\boldsymbol{\theta}}^{\infty} L_T^{\infty},d}$',
                               r'$\varepsilon_{L_T^{\infty}RMS_{\boldsymbol{\theta}},d}$']
        
        fig2, ax2 = plt.subplots(figsize=figSize)
        ax2.set_xlabel(r"Predefined Threshold $\boldsymbol{\delta}_{\text{FSC}}$")
        ax2.set_ylabel(r'$\varepsilon$')

        for normError, normStr in zip(normErrorListExtracted,normErrorStringList):
            ax2.plot(predefTotalErrorList,normError,'*-', label=normStr, markersize = 4)

        ax2.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0,
        )
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_xticks(predefTotalErrorList)
        ax2.grid(True)

        plt.tight_layout()

        if figDir is not None:
            saveFigures(fig2, figName+'Var_'+modelVrOIStr, figDir, dpiVal=dpiVal)
    
def plotTotalErrorCompare(predefTotalErrorList,
                          normErrorSCList,
                          normErrorSCList2,
                          varIdxList,
                          varStrList,
                          figName,
                          figDir=None,
                          figSize=(5,5),
                          dpiVal=300):
    
    
    plt.style.use("./Plotting/dissertation.mplstyle")

    colorList = ['C0','C1','C2','C3','C4','C5']
    
    for modelVarOIIdx, modelVrOIStr in zip(varIdxList, varStrList):
        globalMeanSquareErrorVar_relList = []
        meanSquareErrVar_relList = []
        maxSquareErrorVar_relList = []
        maximumNorm_relList = []
        err_L2theta_LinfT_relList = []
        meanSquareErrVar_relList2 = []
        maxSquareErrorVar_relList2 = []
        maximumNorm_relList2 = []
        err_L2theta_LinfT_relList2 = []
        for i in range(len(predefTotalErrorList)):
            globalMeanSquareErrorVar_relList.append(normErrorSCList[i][1])
            meanSquareErrVar_relList.append(normErrorSCList[i][3][modelVarOIIdx])
            maxSquareErrorVar_relList.append(normErrorSCList[i][5][modelVarOIIdx])
            maximumNorm_relList.append(normErrorSCList[i][7][modelVarOIIdx])
            err_L2theta_LinfT_relList.append(normErrorSCList[i][9][modelVarOIIdx])
            meanSquareErrVar_relList2.append(normErrorSCList2[i][3][modelVarOIIdx])
            maxSquareErrorVar_relList2.append(normErrorSCList2[i][5][modelVarOIIdx])
            maximumNorm_relList2.append(normErrorSCList2[i][7][modelVarOIIdx])
            err_L2theta_LinfT_relList2.append(normErrorSCList2[i][9][modelVarOIIdx])

        normErrorListExtracted = [meanSquareErrVar_relList,
                                  maxSquareErrorVar_relList,
                                  maximumNorm_relList,
                                  err_L2theta_LinfT_relList]
        
        normErrorListExtracted2 = [meanSquareErrVar_relList2,
                                   maxSquareErrorVar_relList2,
                                   maximumNorm_relList2,
                                   err_L2theta_LinfT_relList2]
        
        normErrorStringList = [r'$\varepsilon_{\mathbb{E}L_T^2,d}$',
                               r'$\varepsilon_{L_{\boldsymbol{\theta}}^{\infty}L_T^2,d}$',
                               r'$\varepsilon_{L_{\boldsymbol{\theta}}^{\infty} L_T^{\infty},d}$',
                               r'$\varepsilon_{L_T^{\infty}RMS_{\boldsymbol{\theta}},d}$']
        
        fig2, ax2 = plt.subplots(figsize=figSize)
        ax2.set_xlabel(r"Predefined Threshold $\boldsymbol{\delta}_{\text{FSC}}$")
        ax2.set_ylabel(r'$\varepsilon$')

        idxC = 0
        for normError, normError2, normStr in zip(normErrorListExtracted,normErrorListExtracted2,normErrorStringList):
            ax2.plot(predefTotalErrorList,normError,'*--', color = colorList[idxC], label=normStr+' 0.5', markersize = 4)
            ax2.plot(predefTotalErrorList,normError2,'<-', color = colorList[idxC], label=normStr+' 0.05', markersize = 4)
            idxC +=1

        ax2.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0,
        )
        ax2.set_xticks(predefTotalErrorList)
        ax2.set_xticklabels([str(int(val)) for val in predefTotalErrorList])
        ax2.set_yscale('log')
        ax2.set_xscale('log')
        ax2.grid(True)

        plt.tight_layout()

        if figDir is not None:
            saveFigures(fig2, figName+'Var_'+modelVrOIStr, figDir, dpiVal=dpiVal)

def plotCPoverHarmonics(usedHSetListUnion,
                        nrFSCModels,
                        usedComplexityList,
                        legendTotalErrorList,
                        usedIndexSetUnion,
                        variableList,
                        figName,
                        xTicksStep=1,
                        max_labels = 20,
                        maxNrCPList=None,
                        vertical=False,
                        sameH4all=False,
                        figDir=None,
                        figSize=(8,6),
                        dpiVal=300):
    r"""
    Plot harmonics for each CP, with CPs on the y axis
    and filled boxes for the used harmonics. 
    Includes aFSC models with predefined thresholds. 

    Parameters
    ----------
    usedHSetListUnion : List
        List of the union of used harmonics
    nrFSCModels : int
        Number of aFSC models used in the plot
    usedComplexityList : list
        List of used harmonics for each CP of each aFSC model
    legendTotalErrorList : list
        List of legends for each aFSC model
    usedIndexSetUnion : set
        Unionized set of all employed indices
    variableList : list
        List of str for each variabel in the model 
    figName : str
        The name of the figure file (without extension).
    xTicksStep : int = 1
        Steps for the harmonics to be displayed
    max_labels : int = 20
        Maximum number of y tick labels used for the CPs
    maxNrCPList : list
        List of maximum number of CPs employed in the figure
    vertical : bool
        Boolean to plot the colored boxes vertical instead of horizontal
    sameH4all : bool
        Boolean to use the same plot for all variable
    figDir : str
        The directory where the figure will be saved.
    figSize : tuple
        Figure Size as tuple.
    dpiVal : float
        Resolution of pdf figure in dpi
    """
    
    plt.style.use("./Plotting/dissertation.mplstyle")

    if sameH4all:
        nrVar = 1
    else:
        nrVar = len(usedHSetListUnion)
    
    for idxVar in range(nrVar):
        fig, ax = plt.subplots(figsize=figSize)
        
        # Define subgrid parameters
        n_arrays = nrFSCModels
        if vertical:
            subcols = 1
            subrows = n_arrays
        else:    
            subcols = n_arrays
            subrows = 1
        dx_cell = 1/subcols
        dy_cell = 1/subrows

        # Assign colors from 'tab20' colormap
        colorsTotal = np.array([[0.12156863, 0.46666667, 0.70588235, 1.        ],
                                [0.68235294, 0.78039216, 0.90980392, 1.        ],
                                [1.        , 0.49803922, 0.05490196, 1.        ],
                                [1.        , 0.73333333, 0.47058824, 1.        ],
                                [0.59607843, 0.8745098 , 0.54117647, 1.        ]])
        
        for k in range(n_arrays):
            arrOrg = usedComplexityList[k][idxVar]
            
            if maxNrCPList is not None:
                maxNrCP = maxNrCPList[idxVar]
            else:
                maxNrCP = arrOrg.shape[0]

            arr = arrOrg[:maxNrCP,:]
            rows, cols = np.where(arr == 1)
            
            # Calculate subgrid offsets for this array
            i = k % subcols
            j = k // subcols
            
            # Rectangle offset for this subcell
            x_offset = i * dx_cell
            y_offset = j * dy_cell
            
            for r, c in zip(rows, cols):
                # Rectangle for each 1 in the matrix
                rect = Rectangle(
                    (c + x_offset, r + y_offset),  # (x, y) lower left
                    dx_cell,                      # width
                    dy_cell,                      # height
                    facecolor=colorsTotal[k],
                    edgecolor='k',
                    linewidth=0.5,
                    label=legendTotalErrorList[k] if (r, c) == (rows[0], cols[0]) else None,
                    zorder=3
                )
                ax.add_patch(rect)

        # Style adjustments
        if maxNrCPList is not None:
            yTicks = min(len(usedIndexSetUnion)+1,maxNrCPList[idxVar])
        else:
            yTicks = len(usedIndexSetUnion)+1
        
        xticksArr = np.arange(1, len(usedHSetListUnion[idxVar])+1, 1, dtype=int)
        yticksArr = np.arange(0, yTicks, 1)
        ax.set_xticks(xticksArr)  
        ax.set_yticks(yticksArr)
        ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Harmonics")
        ax.set_ylabel("Collocation Points")

        ax.set_xticklabels([''] * len(xticksArr))
        x_midpoints = list(xticksArr[0:]-0.5) 
        x_labelsOrg = [str(int(ih)) for ih in list(sorted(usedHSetListUnion[idxVar]))]
        
        x_labels = [elem if i % xTicksStep == 0 else '' for i, elem in enumerate(x_labelsOrg)]
        
        ax.set_xticks(x_midpoints, minor=True)
        ax.set_xticklabels(x_labels, minor=True)
        ax.tick_params(axis='x', which='minor', length=0, pad=5)
        
        ax.set_yticklabels([''] * len(yticksArr))
        y_midpoints = list(yticksArr[1:]-0.5) 
        if len(usedIndexSetUnion[0]) == 4:
            y_labels = ['('+str(indexSet[0])+','+str(indexSet[1])+','+str(indexSet[2])+','+str(indexSet[3])+')' for indexSet in usedIndexSetUnion[:maxNrCP]]
        elif len(usedIndexSetUnion[0]) == 1:
            y_labels = ['('+str(indexSet[0])+')' for indexSet in usedIndexSetUnion[:maxNrCP]]
        elif len(usedIndexSetUnion[0]) == 2:
            y_labels = ['('+str(indexSet[0])+','+str(indexSet[1])+')' for indexSet in usedIndexSetUnion[:maxNrCP]]
        
        step = max(1, len(y_labels) // max_labels)
        display_labels = [label if i % step == 0 else '' for i, label in enumerate(y_labels)]

        if maxNrCPList is not None:
            display_labels = display_labels[:maxNrCPList[idxVar]-1]
        
        ax.set_yticks(y_midpoints, minor=True)
        ax.set_yticklabels(display_labels, minor=True)
        ax.tick_params(axis='y', which='minor', length=0)
        
        plt.setp(ax.get_yticklabels(minor=True), rotation=45, ha='right')
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0,
            ncol=1,
            fontsize=8
        )

        if maxNrCPList is not None:
            ax.spines.top.set_visible(False)
        
        plt.tight_layout()
        
        if figDir is not None:
            saveFigures(fig, figName+"_"+variableList[idxVar], figDir, dpiVal=dpiVal)

def plotComplexityOverError(nrModels,
                            errorModelsList,
                            allErrorModelList,
                            legendTotalErrorList,
                            figName,
                            figDir=None,
                            figSize=(5,5),
                            dpiVal=300):
    r"""
    Plot complexity over CP for severel aFSC model with different threshold and
    aFSC models with fixed number of harmonics and CPs.
    
    Parameters
    ----------
    nrModels : int
        Number of aFSC models used in the plot
    errorModelsList : list
        List of used aFSC models with predefined threshold
    allErrorModelList : list
        List of used aFSC modeel with predefine number of harmonics and CP
    allErrorModelLegend : list
        List of legends for each aFSC model
    figName : str
        The name of the figure file (without extension).
    figDir : str
        The directory where the figure will be saved.
    figSize : tuple
        Figure Size as tuple.
    dpiVal : float
        Resolution of pdf figure in dpi
    """

    plt.style.use("./Plotting/dissertation.mplstyle")

    marker_styles = ['o', 'v', '^', '<', '>', 
                        's', 'p', '*', 'h', 'H', 
                        'D', 'd', 'P', 'X', '8',
                        #'+', 'x', '|', '_', '1', '2'
                        ]
    colors = tab20(np.linspace(0, 1, nrModels))
    
    fig, ax = plt.subplots(figsize=figSize)
    ax.set_xlabel(r'$\boldsymbol{\rho}_{\text{FSC}}$')
    ax.set_ylabel(r'Complexity $\sum_{k=1}^{\Lambda} \# I_k$')
    markerIdx = 0
    
    ax.set_xscale('log')
    ax.set_yscale('log')

    # number of each harmonic of each variable of each CP summed
    adaptEr = [np.sum(mod1.errorHBList[-1]) + mod1.errorSCList[-1] for mod1 in errorModelsList]
    adaptComp = []
    for mod2 in errorModelsList:
        totalComplexity2 = 0
        idxAdm = -len(mod2.addedAdmissibleSet)
        if idxAdm == 0:
            idxAdm = len(mod2.harmonicSetListList)

        for harmonicSet2 in mod2.harmonicSetListList[:idxAdm]:
            totalComplexity2 += sum([len(harmSetVar) for harmSetVar in harmonicSet2])
        adaptComp.append(totalComplexity2)
    ax.plot(adaptEr, adaptComp, 'k--', label='Adaptive Error', linewidth = 0.5)
        
    for jIdx, givenModel in enumerate(allErrorModelList):
        actTotalError = np.sum(givenModel.errorHBList[-1]) + givenModel.errorSCList[-1]
        if jIdx < len(errorModelsList):
            printError = rf"{actTotalError:0.1e} $\leq$ "
            totalError = actTotalError
        else:
            printError = ''
            totalError = actTotalError
        totalComplexity = 0
        if hasattr(givenModel,'addedAdmissibleSet'):
            if not givenModel.addedAdmissibleSet == []:
                idxAdm2 = -len(givenModel.addedAdmissibleSet)
            else:
                idxAdm2 = len(givenModel.harmonicSetListList)
        else:
            idxAdm2 = len(givenModel.harmonicSetListList)
        for harmonicSet in givenModel.harmonicSetListList[:idxAdm2]:
            totalComplexity += sum([len(harmonicSetVar) for harmonicSetVar in harmonicSet])
        ax.scatter(totalError,totalComplexity,
                    marker=marker_styles[markerIdx],
                    color=colors[jIdx],
                    s=30,
                    edgecolor='k',
                    linewidth=0.5,
                    label=legendTotalErrorList[jIdx],
                    zorder=3
                )
        markerIdx += 1
        if markerIdx >= len(marker_styles):
            markerIdx = 0

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
        ncol=1,
        fontsize=8
    )
    ax.grid(True)
    ax.minorticks_on()

    plt.tight_layout()

    if figDir is not None:
        saveFigures(fig, figName, figDir, dpiVal=dpiVal)


def plotMeanQuantiles(scMeanPos, 
                      scPosQuantiles,
                      tVec,
                      yLabelStr,
                      xLabelstr,
                      figName,
                      xTicksList=None,
                      figDir=None,
                      figSize=(5,5),
                      dpiVal=300):
    r"""
    Mean, quantiles plot over one period for given variabel.
    
    Parameters
    ----------
    scMeanPos : np.ndarray
        Mean position for the provided variable
    scPosQuantiles : np.ndarray
        2.5 % and 97.5 % quantile of the position for the provided variable
    tVec : np.array
        Array with time points where the mean and quantiles are evaluated
    yLabelStr : str
        String of the y label
    xLabelStr : str
        String of the x label
    figName : str
        The name of the figure file (without extension).
    xTicksList : list
        List with a list of x ticks and a list with corresponding llabels for the x ticks
    figDir : str
        The directory where the figure will be saved.
    figSize : tuple
        Figure Size as tuple.
    dpiVal : float
        Resolution of pdf figure in dpi
    """

    plt.style.use("./Plotting/dissertation.mplstyle")

    fig, ax = plt.subplots(figsize=figSize)

    ax.plot(tVec, scMeanPos.flatten(), '-', color='black',label = r"$\mathbb{E}[x(t,\boldsymbol{\theta})]$")
    ax.plot(tVec, scPosQuantiles[0,:].flatten(), '--', color='red', label = r"$Q_{x(t,\boldsymbol{\theta})}(0.025)$")
    ax.plot(tVec, scPosQuantiles[1,:].flatten(), '--', color='blue', label = r"$Q_{x(t,\boldsymbol{\theta})}(0.975)$")
    ax.set_xlabel(xLabelstr)
    ax.set_ylabel(yLabelStr)
    ax.grid(True)
    if xTicksList is not None:
        ax.set_xticks(xTicksList[0])
        ax.set_xticklabels(xTicksList[1])
    ax.legend(loc='best')
    plt.tight_layout()

    if figDir is not None:
        saveFigures(fig, figName, figDir, dpiVal=dpiVal)

    
def plotDiffMinMax(diffPosMin,
                   diffPosMax,
                   tVec,
                   varStr,
                   xLabelstr,
                   figName,
                   xTicksList=None,
                   figDir=None,
                   yLabelstr=None,
                   figSize=(5,5),
                   dpiVal=300):
    r"""
    Mean, quantiles plot over one period for given variabel.
    
    Parameters
    ----------
    diffPosMin : np.ndarray
        Minimum difference of the position for the provided variable
    diffPosMax : np.ndarray
        Maximum difference of the position for the provided variable
    tVec : np.array
        Array with time points where the mean and quantiles are evaluated
    varStr : str
        String of the used variable
    xLabelStr : str
        String of the x label
    figName : str
        The name of the figure file (without extension).
    xTicksList : list
        List with a list of x ticks and a list with corresponding llabels for the x ticks
    figDir : str
        The directory where the figure will be saved.
    figSize : tuple
        Figure Size as tuple.
    dpiVal : float
        Resolution of pdf figure in dpi
    """

    if yLabelstr is None:
        yLabelstr = r"$\varepsilon^{(i)}=|\widetilde{" + varStr+ r"}^{(i)}-" + varStr+ r"^{(i)}|$"

    plt.style.use("./Plotting/dissertation.mplstyle")

    fig, ax = plt.subplots(figsize=figSize)
    ax.plot(tVec, diffPosMin.flatten(), '-', color='red',label = r"min${}_i (\varepsilon^{(i)})$")
    ax.plot(tVec, diffPosMax.flatten(), '-', color='blue', label = r"max${}_i (\varepsilon^{(i)})$")
    ax.set_xlabel(xLabelstr)
    ax.set_ylabel(yLabelstr)
    ax.set_yscale('log')
    if xTicksList is not None:
        ax.set_xticks(xTicksList[0])
        ax.set_xticklabels(xTicksList[1])
    ax.grid(True)
    ax.legend(loc='best')
    
    plt.tight_layout()

    if figDir is not None:
        saveFigures(fig, figName, figDir, dpiVal=dpiVal)


def plotLimiCycle(meanPos,
                  meanVel,
                  plotSamplesPos,
                  plotSamplesVel,
                  xLabel,
                  yLabel,
                  meanLegendStr,
                  figName,
                  figDir=None,
                  figSize=(5,5),
                  dpiVal=300):
    r"""
    Mean, quantiles plot over one period for given variabel.
    
    Parameters
    ----------
    meanPos : np.ndarray
        Array with the mean position
    meanVel : np.ndarray
        Array with the mean velocity
    plotSamplesPos : np.ndarray
        Array with the sampled position for plotting
    plotSamplesVel : np.ndarray
        Array with the sampled velocity for plotting
    xLabel : str
        String for the x axis label
    yLabel : str
        String for the x axis label
    meanLegendStr : str
        String for legend of mean limit cycle
    figName : str
        The name of the figure file (without extension).
    figDir : str
        The directory where the figure will be saved.
    figSize : tuple
        Figure Size as tuple.
    dpiVal : float
        Resolution of pdf figure in dpi
    """

    plt.style.use("./Plotting/dissertation.mplstyle")

    fig, ax = plt.subplots(figsize=figSize)
    ax.plot(np.append(meanPos,meanPos[0]),
            np.append(meanVel,meanVel[0]),
            '-', color='black',
            label="samples",
            linewidth = 0.8)

    for i in range(plotSamplesPos.shape[0]):
        
        ax.plot(np.append(plotSamplesPos[i,:],plotSamplesPos[i,0]),
                np.append(plotSamplesVel[i,:],plotSamplesVel[i,0]),
                color = "black", linestyle = "-", 
                linewidth = 0.1, alpha = 0.3, )

    ax.plot(np.append(meanPos,meanPos[0]),
            np.append(meanVel,meanVel[0]),
            '-', color='red',
            label=meanLegendStr,
            linewidth = 0.8)
    
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.grid(True)
    ax.legend(loc='best')
    
    plt.tight_layout()

    if figDir is not None:
        saveFigures(fig, figName, figDir, dpiVal=dpiVal)

