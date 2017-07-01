
# coding: utf-8

# In[1]:

import gym
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
import random
import os


# In[2]:

class agent():
    def __init__(self):
        self.LR = 1e-3
        self.env = gym.make("CartPole-v0")
        self.model = self.neural_network_model()
        self.env.reset()
        self.goal_steps = 500
        self.score_requirement = 55
        self.initial_games = 100000
        
        self.np_save_path = 'data_save'
        self.np_file = self.np_save_path + '/data.npy'
        if not os.path.exists(self.np_save_path):
            os.makedirs(self.np_save_path)
            
        self.model_save_path = 'model_save'
        self.model_name = self.model_save_path +'/save'
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        
    def neural_network_model(self):
        keep = 0.8
        input_size = 4
        LR = 1e-3
        network = input_data(shape=[None, input_size, 1], name='input')
        network = fully_connected(network, 128, activation='relu', name = 'hidden_1')
        network = dropout(network,keep)
        network = fully_connected(network, 256, activation='relu', name = 'hidden_2')
        network = dropout(network,keep)
        network = fully_connected(network, 512, activation='relu', name = 'hidden_3')
        network = dropout(network,keep)
        network = fully_connected(network, 256, activation='relu', name = 'hidden_4')
        network = dropout(network,keep)
        network = fully_connected(network, 128, activation='relu', name = 'hidden_5')
        network = dropout(network,keep)
        network = fully_connected(network, 2, activation='softmax', name = 'softmax')
        network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
        model = tflearn.DNN(network, tensorboard_verbose=3)
        return model
        
    def generate_data(self):
        training_data = []
        scores = []
        accepted_scores = []
        for _ in range(self.initial_games):
            score = 0
            game_memory = []
            prev_observation = []
            for _ in range(self.goal_steps):
                action = random.randrange(0,2)
                observation, reward, done, info = self.env.step(action)

                if len(prev_observation) > 0 :
                    game_memory.append([prev_observation, action])
                prev_observation = observation
                score+=reward
                if done: break

            if score >= self.score_requirement:
                accepted_scores.append(score)
                for data in game_memory:
                    if data[1] == 1:
                        output = [0,1]
                    elif data[1] == 0:
                        output = [1,0]

                    training_data.append([data[0], output])

            self.env.reset()
            scores.append(score)

        training_data_save = np.array(training_data)
        np.save(self.np_file ,training_data_save)
        print('Average accepted score:',mean(accepted_scores))
        print('Median score for accepted scores:',median(accepted_scores))
        print(Counter(accepted_scores))

        return training_data
    
    def train_model(self, model=False):
        training_data = np.load(self.np_file)
        X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
        y = [i[1] for i in training_data]
        keep = 0.8
        self.model.fit({'input': X}, {'targets': y}, n_epoch=1, snapshot_step=500, show_metric=True, run_id='openai_learning')
        self.model.save(self.model_name)
        return model
    
    def play(self, render = False, num = 100, load = False):
        scores = []
        choices = []
        if load:
            self.model.load(self.model_name)
        for each_game in range(num):
            score = 0
            game_memory = []
            prev_obs = []
            self.env.reset()
            for _ in range(self.goal_steps):
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
    
    def run(self):
        self.generate_data()
        self.train_model()
        self.play()


# In[3]:
if __name__ == '__main__':
	more = agent()
	more.run()
	more.play(load = True, render= True)

