import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
LR = 1e-3






def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network,.8)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network,.8)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network,.8)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network,.8)
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network,.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='/tmp/tensorflow_logs/lll')

    return model