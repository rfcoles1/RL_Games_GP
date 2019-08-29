import numpy as np
import gym
import matplotlib.pyplot as plt

import time
from Tkinter import *
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


network = Network(config)
weights = np.load('../../../Genetic_Algorithms/models/GP_Water-v0/126.npz')
network.w_in = weights['w_in']
network.w_hidden = weights['w_h']
network.w_out = weights['w_out']

env = gym.make('GP_Water-v0')
s = env.reset()
print(env.unwrapped.TargetTemp, env.unwrapped.TargetMass)
def plt_model(ind, s, r):   
        minT = -50
        maxT = 150
        minE = env.unwrapped.engine.get_Energy_From_Temp(minT)
        maxE = env.unwrapped.engine.get_Energy_From_Temp(maxT)
        
        Energy = np.linspace(minE,maxE, 100).reshape(-1,1)
        TempMeans = np.zeros(len(Energy))
        TempSdvs = np.zeros(len(Energy))
        Temp = np.zeros(len(Energy))
        MassMeans = np.zeros(len(Energy))
        MassSdvs = np.zeros(len(Energy))
        Mass = np.zeros(len(Energy))

        for i in range(len(Energy)):
            out = env.unwrapped.engine.TempModel._raw_predict(np.array([Energy[i]]).reshape(-1,1))
            TempMeans[i] = out[0].flatten()[0]
            TempSdvs[i] = out[1].flatten()[0]
            Temp[i] = env.unwrapped.engine.get_true_value(Energy[i])[0]

            out = env.unwrapped.engine.MassModel._raw_predict(np.array([Energy[i]]).reshape(-1,1))
            MassMeans[i] = out[0].flatten()[0]
            MassSdvs[i] = out[1].flatten()[0]
            Mass[i] = env.unwrapped.engine.encodeMass(env.unwrapped.engine.get_true_value(Energy[i])[1])

        plt.figure()
        plt.subplot(121)
        plt.plot(Energy, Temp, 'k', label = 'True')
        plt.plot(Energy, TempMeans, label = 'Mean')
        plt.fill_between(Energy.flatten(), TempMeans - TempSdvs, TempMeans + TempSdvs, facecolor = 'r', alpha = 0.5, label = '1 sigma range')
        plt.ylabel('Temperature ($^\circ$C)')
        plt.xlabel('Energy Added (kJ)')
        plt.ylim(top = max(Temp) +10, bottom = min(Temp) - 10)
        plt.legend()

        plt.subplot(122)
        plt.plot(Energy, Mass, 'k', label = 'True')
        plt.plot(Energy, MassMeans, label = 'Mean')
        plt.fill_between(Energy.flatten(), MassMeans - MassSdvs, MassMeans + MassSdvs, facecolor = 'r', alpha = 0.5, label = '1 sigma range')
        plt.ylabel('MassFractions')
        plt.xlabel('Energy Added (kJ)')
        plt.ylim(top = max(Mass) + 0.1, bottom = min(Mass) - 0.1)
        plt.legend()

        plt.tight_layout()


i = 0
while True:
    a = network.predict(s.flatten())
    a = np.argmax(a)
    s, reward, done, _, = env.step(a)

    plt_model(i,s,reward)
    print(i, a, s, reward)
    i += 1
    if done:
        break

    time.sleep(0.5)




