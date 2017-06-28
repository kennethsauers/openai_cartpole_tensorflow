
# coding: utf-8

# In[1]:

import gym
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import model
import random
goal_steps = 500
initial_games = 10000


# In[2]:

class agents():
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.model = model.neural_network_model(4)
        #self.model.load('model_save/test')
    def play(self, render = False, num = 100):
        scores = []
        choices = []
        for each_game in range(num):
            score = 0
            game_memory = []
            prev_obs = []
            self.env.reset()
            for _ in range(goal_steps):
                if render:
                    self.env.render()

                if len(prev_obs)==0:
                    action = random.randrange(0,2)
                else:
                    action = np.argmax(self.model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

                choices.append(action)

                new_observation, reward, done, info = self.env.step(action)
                prev_obs = new_observation
                game_memory.append([new_observation, action])
                score+=reward
                if done: break

            scores.append(score)

        print('Average Score:',sum(scores)/len(scores))
        print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))


# In[3]:

if __name__ == '__main__':
    n = agents()
    n.play()


# In[ ]:



