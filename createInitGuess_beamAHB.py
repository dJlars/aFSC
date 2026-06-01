import os
import dill

from beamEulerBernoulli.eulerBernoulliaHB import eulerBernoulliAHB

from generalUtils.argparser import get_config_Beam

from generalUtils.filemanager import save_script
saveResultsPath = "./"

result_folder = save_script(saveResultsPath,
                            os.path.realpath(__file__),
                            "beamEB","", 
                            max_daily_folders = 7, 
                            max_res_folders = 7)

configHB = get_config_Beam("beamEulerBernoulli/beamEBHB.yaml", result_folder)


with open("beamEulerBernoulli/deflationFakeSolution_MeanVal_Small_Middle.pkl","rb") as f:
    deflationFakesol = dill.load(f)

myAHBbeamEB = eulerBernoulliAHB([{1,2,3,4}],
                                configHB,
                                deflation=True,
                                # maxDefSol = 3,
                                deflationFakeSolutions = deflationFakesol,
                                pointForce=True,
                                nonAdaptive=False, # change 
                                onlyAdd = True,
                                RootMethod = ['hybr'],
                                getNewInitialGuess = True,
                                AbsRangeRndGuess = [-1e-4,1e-4],
                                usedSamples = 0.1,
                                )

initSolList = myAHBbeamEB.getInitialGuess(0)

mySolFourierCoeff = myAHBbeamEB.adaptiveHB(initSolList)


with open("beamEulerBernoulli/initGuessForBeam.pkl","rb") as f:
    a_testVar = dill.load(f)

# store results in lists according to needed format for initial guess
initList = []
for sol, harmSet in zip(myAHBbeamEB.deflationSolutionList,
                        myAHBbeamEB.deflationHarmonicSetList):
    
    allSolList = []
    idxCur = 0

    for idxDoF, dofHarm in enumerate(harmSet):
        
        for idxH, harm in enumerate(dofHarm):
            if idxH == 0: 
                dofCoeffList = [sol[idxCur]]
                idxCur +=1
            
            dofCoeffList.append(sol[idxCur:idxCur+2])
            idxCur += 2
        allSolList.append(dofCoeffList)
    
    initList.append(allSolList)

with open("initGuessForBeam.pkl","wb") as f:
    dill.dump(initList,f)
        


print("This is the way!")