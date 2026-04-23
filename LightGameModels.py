from LightGameModelRoot import LightGameModelRt
import numpy as np
import gymnasium as gym
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from pathlib import Path
from TestModels import LunarTestModel, BipedalTestModel, HumanoidTestModel, BipedalTQCTestModel


class LunarGameModel(LightGameModelRt):
    def __init__(self, trainData=True):
        super()
        self.lightModel =  MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=256)
        self.gymEnv = gym.make("LunarLander-v3", continuous=True)
        self.modelName = "LightLunarModelTest"
        self.largeModelName = "LargeLunarModel"
        self.pcaTransformer = PCA()
        self.applyPCA = False
        self.stateCount=8
        if trainData:
            self.loadModel()

    def getLargeModel(self):
        largeModel = LunarTestModel(gymEnv=self.gymEnv, modelName=self.largeModelName)
        return largeModel
    
    def predict(self, state):
        return np.clip(super().predict(state), -1.0, 1.0)
    
class BipedalGameModel(LightGameModelRt):
    def __init__(self, trainData=True):
        super()
        self.lightModel =  MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=512)
        self.gymEnv = gym.make("BipedalWalker-v3")
        self.modelName = "LightBipedalModelTest"
        self.largeModelName = "LargeBipedalModel"
        self.pcaTransformer = PCA()
        self.applyPCA = True
        self.stateCount=24
        if trainData:
            self.loadModel()

    def getLargeModel(self):
        largeModel = BipedalTestModel(gymEnv=self.gymEnv, modelName=self.largeModelName)
        return largeModel
    
class BipedalTQCGameModel(LightGameModelRt):
    def __init__(self, trainData=True):
        super()
        self.lightModel =  MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=512)
        self.gymEnv = gym.make("BipedalWalker-v3")
        self.modelName = "LightBipedalHardModel"
        self.largeModelName = "LargeBipedalHardcore"
        self.pcaTransformer = PCA()
        self.applyPCA = True
        self.stateCount=24
        self.anchorCount=3000
        if trainData:
            self.loadModel()

    def getLargeModel(self):
        largeModel = BipedalTQCTestModel(gymEnv=self.gymEnv, modelName=self.largeModelName)
        return largeModel
    
class HumanoidGameModel(LightGameModelRt):
    def __init__(self, trainData=True):
        super()
        self.lightModel =  MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=512)
        self.gymEnv = gym.make("Humanoid-v4")
        self.modelName = "LightHumanoidModel"
        self.largeModelName = "LargeHumanoidModel"
        self.pcaTransformer = PCA()
        self.applyPCA = True
        self.stateCount=376
        if trainData:
            self.loadModel()

    def getLargeModel(self):
        largeModel = HumanoidTestModel(gymEnv=self.gymEnv, modelName=self.largeModelName)
        return largeModel