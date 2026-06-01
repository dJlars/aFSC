# adaptive Fourier Stochatic Collocation

This repository contains a framework for the adaptive Fourier Stochatic Collocation (aFSC) method. The method combines the harmonic balance (HB) with an adaptive stochastic collocation method to quantify uncertainties in limit cycle oscillations of nonlinear dynmaical systems.

The fundamental principal of the aFSC method is to compute teh solutiona of an adaptive HB at each collocation point. The selection of the collocation point and number of harmonics are determined by an derived error measure.

The framework is applied to the Duffing oscillator with a cubic and sine term as a benchmark example and to the model of the electrophysiology of the $\beta$ cell. Since the IGA algorithm is an in-house development that is planned to be published separately in the future, the the nonlinear Euler-Bernuolli beam is not included in the provided repository .

The relevant parameters are the threshold for the error measure as well as the balancing factor.

Further details are given in the publication [tbd].

# Folder Structure

The folder ``aFSC`` contains the classes for the aFSC method. Most importantly the ``fourierStochasticCollocation.py`` for the collocation method and the ``adaptiveHB.py`` for the adaptive HB.
The folder ``generalUtils`` contains utility functions. The ``mainHelperFunctions.py`` contains functions for evaluating the determined aFSC surrogates. The remaining files are support files for parsing, file managing and logger.

Each example has its own folder containing the child class from the aHB method and a configuration ``.yaml`` file. Additionally one data folder where the calculated aFSC models are stored as well as result foldedr where the executed file is saved.

The main files for the example are in the root directory. After running ``main[[example]].py``, the corresponding plotting file in the folder ``Plotting`` can be executed. The ``plottingaFSC.py`` file contains all fucntions to create the plots. The example files handle the data distribution towards the plotting functions.

#  Getting Started

All necessary packages are included in the ``requirements.txt`` file. It is advised that a virtual environment should be installed using the pip package manager, and that the requisite packages be installed subsequently. 

# Basic Program Structure

The main file is structured in the following sections:
- Setup
- aFSC model & predefined surrogates
- Complexity calculation
- MC and aFSC sampling
- SC and HB convergence
- Saving data

The plotting file loads the relevant data and plots the requested figures.

**Setup**

In this section all essential parameters are configured to regulate the algorithm's behavior, e.g. including the initiation of a convergence study.
In addition, one ``.yaml`` file is read, which contain all information pertinent to the system. For parameters that require specific attention, please refer to the ``.yaml`` files of the provided examples.

**aFSC model & predefined surrogates**

Each main has a construction function to create the surrogate. Here problem specific data is provided. Adaptive aFSC models are created followed by predefined surrogates.
Both determined models are utilized to determine their complexity.

**MC and aFSC sampling**

To demonstrate the precision of the aFSC, a Monte Carlo (MC) sampling is conducted. 
Initialization of the model, which is used for the MC study, is the first step, followed by the evaluation for the provided sampels.
Subsequently, the adaptive surrogates are  evaluated for the same sample set and the error norms are calculated.


**SC and HB convergence**

For both the frequency and uncertain domain convergence studies are conducted. The error is determined via the error norms from before.


**Saving data**

After generating the data, all requiered data is saved for later plotting

**Plotting**

The ploting file loads the data and creates the corresponding plots. Here some values, sucha s the number of samples, need to be set to load the correct data.  
The following functions for the plots are provided:
- Mean limit cycle for selected variables with some samples
- Mean, 0.025 and 0.975 qunatiles of the variable position over time
- Minimum and maximum difference at each time point compared to the MC samples
- SC and HB convergence study for selected variables
- convergence of error for adaptive surrogates
- If applicable, sparse grid cuts to illustrate anisotropy
- Used collocation points over harmonics to visualize sparse pattern
- Complexity over derived error measure to demonstrate efficient choice of harmonics and collocation points


# Building Your Own FgPC System

All examples can be used as templates, depending on the problem at hand. If it is self-excited, then the cell biology problem is appropriate. Mostly problem related functions like ``calculateElectricResiduum`` in the cell biology example have to be replaced.

