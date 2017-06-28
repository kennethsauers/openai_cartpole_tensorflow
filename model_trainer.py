
# coding: utf-8

# In[1]:

import numpy as np
import model


# In[2]:

def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input': X}, {'targets': y}, n_epoch=1, snapshot_step=500, show_metric=True, run_id='openai_learning')
    model.save('model_save/test')
    return model


# In[3]:

if __name__ == '__main__':
    training_data = np.load('saved.npy')
    model = model.neural_network_model(4)
    model = train_model(training_data, model=model)

