
# coding: utf-8

# In[1]:

import gym
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import random


# In[2]:

import model
import model_trainer
import data_generator
import agent


# In[3]:

if __name__ == '__main__':
    data = data_generator.enviroment()
    data.initial_games = 10000
    data.score_requirement = 50
    x = data.generate_data()
    training_data = np.load('saved.npy')
    model = model.neural_network_model(4)
    model = model_trainer.train_model(training_data, model=model)
    player = agent.agents()
    player.model = model
    player.play()

