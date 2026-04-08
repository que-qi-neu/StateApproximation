from AchorVectors import AnchorVectors
from sklearn.neighbors import KDTree
import joblib
from pathlib import Path
from BaseModel import BaseModel
from abc import abstractmethod
import numpy as np
import gymnasium as gym
import sklearn

sklearn.set_config(assume_finite=True)

# This is the general parent class for all Game Models
# Child class will specify game environment, how many states it contains, what model will be used for training and etc
class LightGameModelRt(BaseModel):
    stateVectorTree = None
    lightModel = None
    gymEnv = None
    stateList = None
    modelName = None
    largeModelName = None
    applyPCA = False
    pcaTransformer = None
    savedPath = None
    anchorCount = 1024
    stateCount = 0
    def __init__(self, trainData=True):
        pass

    def loadModel(self):
        self.savedPath = Path(f"model/{self.modelName}.pkl")    
        if self.savedPath.exists():
            model = joblib.load(self.savedPath)
            self.__dict__.update(model.__dict__)
        else:
            self.train()

    # The training will generate a small model which will be used for state vector approximation
    def train(self):
        # Get the all states and much smaller anchor states
        anchorVecObj = AnchorVectors(model=self.getLargeModel(), env=self.gymEnv)
        self.anchorVectors, rawStates, rawActions, self.pcaTransformer = anchorVecObj.generateAnchors(anchorCount=self.anchorCount,
                                                                                                       epoch=1000, applyPCA=self.applyPCA)
        print("anchors: ", len(self.anchorVectors), "raw vectors: ", len(rawStates))
        stateCount = self.stateCount
        # The PCA is used for shrinking states
        if self.applyPCA:
            stateCount = self.pcaTransformer.n_components_
            print("state dimensions after compression: ", stateCount)
        self.anchorStatesVectors = self.anchorVectors[:,:stateCount]
        self.anchorActionVectors = self.anchorVectors[:,stateCount:]
        # saves all anchor states into this search-optimized tree
        self.stateVectorTree = KDTree(self.anchorStatesVectors, leaf_size=50)


        # now find all the distances between anchor states and entire states
        distances, indices = self.stateVectorTree.query(rawStates, k=1)
        indices = indices.flatten()
        stateDiffVectors = rawStates - self.anchorStatesVectors[indices]
        actionDiffVectors = rawActions - self.anchorActionVectors[indices]

        # train the model, so that it gives prediction of action change given the vector 
        # difference between the input state and the anchor state
        self.lightModel.fit(stateDiffVectors, actionDiffVectors)
        # Save trained model
        joblib.dump(self, self.savedPath)

    def predict(self, state:list[float]):
        state2D = np.atleast_2d(state)
        # If we used PCA to shrink state vectors during training, shrink the input state as well
        if self.applyPCA:
            state2D = self.pcaTransformer.transform(state2D)
        distances, indices = self.stateVectorTree.query(state2D , k=1)
        indices = indices.flatten()

        anchorState = self.anchorStatesVectors[indices]
        anchorState2D = np.atleast_2d(anchorState)
        anchorAction = self.anchorActionVectors[indices]
        # Predict based on the difference between input state and closest anchor state
        actionDiff = self.lightModel.predict(state2D - anchorState2D)
        result = anchorAction + actionDiff

        if np.array(state).ndim == 1:
            return result[0] 
            
        return result
    
    def getStateCount(self):
        return self.stateCount 
    
    @abstractmethod
    def getLargeModel():
        return None



