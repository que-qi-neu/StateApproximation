from BaseModel import BaseModel
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import TQC
from pathlib import Path

import gymnasium as gym
import numpy as np
import sys

sys.modules["gym"] = gym

class LunarTestModel(BaseModel):
    def __init__(self, gymEnv, modelName:str, trainingSteps=500000):    
        path = Path(f"model/{modelName}.zip")    
        self.env = gymEnv
        self.stateCount = 8
        if path.exists():
            print('Loading the model: ', path)
            self.model = SAC.load(path=path, env=self.env)
        else:
            self.model = SAC("MlpPolicy", self.env, verbose=1)
            self.model.learn(total_timesteps=trainingSteps)
            self.model.save(path=path)

    def predict(self, input):
        result, placeHolder = self.model.predict(observation=input, deterministic=False)
        return result
    def getStateCount(self):
        return self.stateCount
    

class BipedalTestModel(BaseModel):
    def __init__(self, gymEnv, modelName:str):    
        path = Path(f"model/{modelName}.zip")    
        self.env = gymEnv
        self.stateCount = 24
        if path.exists():
            print('Loading the model: ', path)
            self.model = SAC.load(path=path)
            self.model.set_env(gymEnv)

    def predict(self, input):
        result, placeHolder = self.model.predict(observation=input, deterministic=False)
        return result
    def getStateCount(self):
        return self.stateCount
    
class BipedalTQCTestModel(BaseModel):
    def __init__(self, gymEnv, modelName:str):    
        path = Path(f"model/{modelName}.zip")    
        self.env = gymEnv
        self.stateCount = 24
        if path.exists():
            print('Loading the model: ', path)
            custom_objects = {
                "observation_space": gymEnv.observation_space,
                "action_space": gymEnv.action_space,
            }
            self.model = TQC.load(path=path, custom_objects=custom_objects)
            self.model.set_env(gymEnv)

    def predict(self, input):
        result, placeHolder = self.model.predict(observation=input, deterministic=False)
        return result
    def getStateCount(self):
        return self.stateCount
    
class HumanoidTestModel(BaseModel):
    def __init__(self, gymEnv, modelName:str):    
        path = Path(f"model/{modelName}.zip")    
        self.env = gymEnv
        if path.exists():
            print('Loading the model: ', path)
            self.model = TQC.load(path=path)
            self.model.set_env(gymEnv)

    def predict(self, input):
        result, placeHolder = self.model.predict(observation=input, deterministic=False)
        return result