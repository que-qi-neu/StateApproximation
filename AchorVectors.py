import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from pathlib import Path
from BaseModel import BaseModel

# In general, machine learning in game play is very expensive and we can't frequently pass the data to large machine learning
# model and wait for result. This algorithm will first use a large machine learning to produce a very light
# machine learning model that handles game micro very well. 
class AnchorVectors:
    anchorCounts = 2048
    anchorVectorsList=[]
    def __init__(self, model:BaseModel, env):
        self.model = model
        self.env = env

    # Using deep learning to make predictions and generate a number of states and actions stored in list as vectors
    # This list will be later processed to greately reduce the vector counts and only leave useful vectors
    def generateRawAnchors(self, stateDimension=0.9999, applyPCA=False, epoch=512):

        rawStateVectors=[]
        rawActionVectors=[]

        # Make certain amount of games to get a list of actions and states
        for epoch in range(epoch):

            done = False
            truncate = False
            observation, info = self.env.reset()
            while not(done or truncate):
                actionPre = self.model.predict(input=observation)
                actionPreVec = np.array(actionPre)
                statePreVec = np.array(observation)

                rawStateVectors.append(statePreVec)
                rawActionVectors.append(actionPreVec)

                observation, reward, done, truncate, info= self.env.step(actionPre)
        
        pcaTransformer = PCA(n_components=stateDimension)
        if applyPCA:
            rawStateVectors = pcaTransformer.fit_transform(rawStateVectors)

        return np.array(rawStateVectors), np.array(rawActionVectors), pcaTransformer
    
    # Using k-mean vector clustering to get the averages of vectors from each clusters. These highly frequent 
    # and representative vectors will be used as anchors for next phase.
    def generateAnchors(self, anchorCount, epoch=512, applyPCA=False):
        self.anchorCounts = anchorCount
        # self.anchorCounts = int(sum(stateList) * sum(stateList) / len(stateList)) 

        rawStates, rawActions, pcaTransformer = self.generateRawAnchors( applyPCA=applyPCA, epoch=epoch)
        mergedRawVectors = np.hstack((rawStates, rawActions))

        kmeans = MiniBatchKMeans(n_clusters=self.anchorCounts, batch_size=10000,n_init="auto")
        kmeans.fit(mergedRawVectors)
        centroids = kmeans.cluster_centers_

        # print(centroids[:3], 'Count: ', len(centroids), sep='\n')

        return centroids, rawStates, rawActions, pcaTransformer



# vec = AnchorVectors()
# vec.generateAnchors(stateList=[1,2,3,4,5,6,7,8], epoch=30)


