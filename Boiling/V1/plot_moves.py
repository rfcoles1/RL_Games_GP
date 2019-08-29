import numpy as np
from scipy import stats
import GPy 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatch

import gym
import os 
import sys
sys.path.insert(0,'../../..')
import GP_Games

sys.path.insert(0,'../../../Genetic_Algorithms')
from GA_Config import Config
from GA_Network import Network

config = Config()
config.num_layers = 2
config.num_hidden = 128
config.env_name = 'GP_Water-v0'
config.a_size = 4

network = Network(config)

generations = 100
steps = 100
iters = 1
actions = np.zeros([steps, generations, iters])

TargetTemp = 100
TargetMass = [0,0.5,0.5]
StartTemp = 21
StartMF = [0,1,0]

for gen in range(1,generations):
    weights = np.load('../../../Genetic_Algorithms/models/GP_Water-v0/' + str(gen) + '.npz')
    network.w_in = weights['w_in']
    network.w_hidden = weights['w_h']
    network.w_out = weights['w_out']
    print gen

    curr_actions = np.zeros(iters)
    for j in range(iters):
        env = gym.make(config.env_name)
        s = env.reset() 
        env.unwrapped.engine.set_target(TargetTemp, TargetMass)
        env.unwrapped.engine.T = StartTemp
        env.unwrapped.engine.MassFractions = StartMF
        env.unwrapped.engine.M = env.unwrapped.engine.encodeMass(StartMF)
        env.unwrapped.engine.EnergyIn = env.unwrapped.engine.get_Energy_From_Temp(StartTemp)
        env.unwrapped.engine.input_memory = []
        env.unwrapped.engine.output_memory_T = []
        env.unwrapped.engine.output_memory_M = []
        env.unwrapped.engine.add_data() 
        s = env.unwrapped.get_state()

        done = False
        i = 0
        while done == False:
            a = network.predict(s.flatten())
            a = np.argmax(a)
            s, reward, done, _ = env.step(a)
            actions[i,gen-1,j] = a
            i += 1

A = stats.mode(actions, axis = 2)[0].reshape([steps, generations])

plt.figure()
im = plt.imshow(A)
plt.xlabel('Generation')
plt.ylabel('Step Number') 
plt.xticks(np.arange(0, generations+1, 25))
plt.yticks(np.arange(0, steps+1, 25))

colors = [im.cmap(im.norm(value)) for value in range(config.a_size)]
labels = ['Action 0 - Subtract Energy', 'Action 1 - Wait', 'Action 2 - Add Energy', 'Action 3 - Observe']
patches = [mpatch.Patch(color=colors[value],label = labels[value]) for value in range(config.a_size)] 
plt.legend(handles=patches, bbox_to_anchor=(1.05,1), loc=2, borderaxespad=0.)
plt.grid(True)
