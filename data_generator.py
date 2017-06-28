
# coding: utf-8

# In[1]:

import gym
import numpy as np
import random
from statistics import median, mean
from collections import Counter


# In[2]:

class enviroment():
    def __init__(self):
        self.LR = 1e-3
        self.env = gym.make("CartPole-v0")
        self.env.reset()
        self.goal_steps = 500
        self.score_requirement = 25
        self.initial_games = 1000
    def generate_data(self):
        # [OBS, MOVES]
        training_data = []
        # all scores:
        scores = []
        # just the scores that met our threshold:
        accepted_scores = []
        # iterate through however many games we want:
        for _ in range(self.initial_games):
            score = 0
            # moves specifically from this environment:
            game_memory = []
            # previous observation that we saw
            prev_observation = []
            # for each frame in 200
            for _ in range(self.goal_steps):
                # choose random action (0 or 1)
                action = random.randrange(0,2)
                # do it!
                observation, reward, done, info = self.env.step(action)

                # notice that the observation is returned FROM the action
                # so we'll store the previous observation here, pairing
                # the prev observation to the action we'll take.
                if len(prev_observation) > 0 :
                    game_memory.append([prev_observation, action])
                prev_observation = observation
                score+=reward
                if done: break

            # IF our score is higher than our threshold, we'd like to save
            # every move we made
            # NOTE the reinforcement methodology here. 
            # all we're doing is reinforcing the score, we're not trying 
            # to influence the machine in any way as to HOW that score is 
            # reached.
            if score >= self.score_requirement:
                accepted_scores.append(score)
                for data in game_memory:
                    # convert to one-hot (this is the output layer for our neural network)
                    if data[1] == 1:
                        output = [0,1]
                    elif data[1] == 0:
                        output = [1,0]

                    # saving our training data
                    training_data.append([data[0], output])

            # reset env to play again
            self.env.reset()
            # save overall scores
            scores.append(score)

        # just in case you wanted to reference later
        training_data_save = np.array(training_data)
        np.save('saved.npy',training_data_save)

        # some stats here, to further illustrate the neural network magic!
        print('Average accepted score:',mean(accepted_scores))
        print('Median score for accepted scores:',median(accepted_scores))
        print(Counter(accepted_scores))

        return training_data


# In[3]:

if __name__ == '__main__':
    model = enviroment()
    model.initial_games = 100000
    model.score_requirement = 50
    x = model.generate_data()

