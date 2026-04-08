from LightGameModels import LunarGameModel, BipedalGameModel, HumanoidGameModel, BipedalTQCGameModel
from TestModels import *
from LightGameModelRoot import LightGameModelRt
from BaseModel import BaseModel
import numpy as np
import gymnasium as gym
import time


def evaluateModel(testEnv:gym.Env, model:BaseModel, batch:int, epochCount = 500, printStats=True):
    truncateCount = 0
    doneCount = 0
    rewardList = []
    predictionTimes = []
    start = time.perf_counter()
    observations, info = testEnv.reset()
    currentRewards = np.zeros(batch)

    while len(rewardList) < epochCount:
        start = time.perf_counter()
        actions = model.predict(observations)

        end = time.perf_counter()
        predictionTimes.append(end - start)
        observations, rewards, done, truncate, info = testEnv.step(actions)
        currentRewards += rewards
        finish = np.logical_or(done, truncate)
        if np.any(finish):
            rewardList.extend(currentRewards[finish])
            currentRewards[finish] = 0

    if printStats:
        start = time.perf_counter()
        randomInputs = np.random.rand(100000, model.getStateCount()).astype(np.float32)
        model.predict(randomInputs)
        end = time.perf_counter()
        print('Average prediction Time: ', sum(predictionTimes) / len(predictionTimes))
        # print('Total Runtime: ', totalEnd - totalStart)
        # print('done: ', doneCount)
        # print('truncated: ', truncateCount)
        print('first few rewards: ', rewardList[:10])
        print('rewards average: ', sum(rewardList) / len(rewardList))
        print('\n')


def compareModels(envName:str, model:LightGameModelRt, kwargs):
    env = gym.make(envName, **kwargs)
    testModel = model.getLargeModel()

    envDisplay = gym.make(envName, render_mode="human", **kwargs)
    evaluateModel(testEnv=envDisplay, model=model, batch=1, epochCount=3, printStats=False)
    envDisplay.close()

    batch = 10
    envVec = gym.make_vec(envName, num_envs=batch, **kwargs)

    print('\nPrinting result from project model\n')
    evaluateModel(testEnv=envVec, model=model, batch=batch)
    print('\nNow printing benchmark model\n')
    evaluateModel(testEnv=envVec, model=testModel, batch=batch)
    env.close()
    envVec.close()


print('Showing Lunar Lander')
compareModels(envName="LunarLander-v3", model=LunarGameModel(), kwargs={"continuous": True})
print('Showing Bipedal Walker')
compareModels(envName="BipedalWalker-v3", model=BipedalGameModel(), kwargs={})
