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
weights = np.load('../../../Genetic_Algorithms/models/GP_Water-v0_THIS/30.npz')
network.w_in = weights['w_in']
network.w_hidden = weights['w_h']
network.w_out = weights['w_out']

class MyLoop():
    def __init__(self, root):
        self.running = True
        self.quit = False
        self.root = root
        self.root.bind("<space>", self.switch)
        self.root.bind("<Escape>", self.exit)
        self.root.bind("<m>", self.changeTarget)
    
        self.env = gym.make('GP_Water-v0')
        s = self.env.reset()
        i = 0
        
        self.fig, self.ax = plt.subplots(1,2)
        plt.ion()
        self.fig.show()
        self.plt_model(i,s,0)

        raw_input('PRESS ENTER TO CONTINUE.')

        while not self.quit:
            self.root.update()
            if self.running:
                a = network.predict(s.flatten())
                a = np.argmax(a)
                s, reward, done, _, = self.env.step(a)
    
                self.plt_model(i,s,reward)
                print(i, a, s, reward)
                i += 1
                time.sleep(0.5)
            else:
                time.sleep(0.1) 
                
    def switch(self, event):
        print(['Unpaused', 'Paused'][self.running])
        self.running = not(self.running)

    def exit(self, event):
        self.quit = True
        self.root.destroy()

    def changeTarget(self,event):
        inp = np.fromstring(raw_input("Please enter a new target\n"), dtype=float, sep = ' ')
        if len(inp) == 1:
            print('Setting Temp to %f' %(inp[0]))
            tmp = self.env.unwrapped.engine.get_Energy_From_Temp(inp[0])
            tmp = self.env.unwrapped.engine.get_true_value(tmp)
            self.env.unwrapped.engine.set_target(tmp[0], tmp[1])
        #if len(a) == 3:
        #    print('Setting Mass Fraction to [%f, %f, %f]' %(a[0], a[1], a[2]))
        else:
            print('nope')

    def plt_model(self, ind, s, r):   
        minT = -50
        maxT = 150
        minE = self.env.unwrapped.engine.get_Energy_From_Temp(minT)
        maxE = self.env.unwrapped.engine.get_Energy_From_Temp(maxT)
        
        Energy = np.linspace(minE,maxE, 100).reshape(-1,1)
        TempMeans = np.zeros(len(Energy))
        TempSdvs = np.zeros(len(Energy))
        Temp = np.zeros(len(Energy))
        MassMeans = np.zeros(len(Energy))
        MassSdvs = np.zeros(len(Energy))
        Mass = np.zeros(len(Energy))

        for i in range(len(Energy)):
            out = self.env.unwrapped.engine.TempModel._raw_predict(np.array([Energy[i]]).reshape(-1,1))
            TempMeans[i] = out[0].flatten()[0]
            TempSdvs[i] = out[1].flatten()[0]
            Temp[i] = self.env.unwrapped.engine.get_true_value(Energy[i])[0]

            out = self.env.unwrapped.engine.MassModel._raw_predict(np.array([Energy[i]]).reshape(-1,1))
            MassMeans[i] = out[0].flatten()[0]
            MassSdvs[i] = out[1].flatten()[0]
            Mass[i] = self.env.unwrapped.engine.encodeMass(self.env.unwrapped.engine.get_true_value(Energy[i])[1])
       
        
        plt.subplot(211)
        plt.cla()
        plt.plot(Energy, Temp, 'k', label = 'True')
        plt.plot(Energy, TempMeans, 'r', label = 'Mean')
        plt.fill_between(Energy.flatten(), TempMeans - TempSdvs, TempMeans + TempSdvs, facecolor = 'r', alpha = 0.5, label = '1 sigma range')
        plt.scatter(self.env.unwrapped.engine.EnergyIn, self.env.unwrapped.engine.get_state()[0][0])
        plt.ylabel('Temperature ($^\circ$C)')
        plt.xlabel('Energy Added (kJ)')
        plt.ylim(top = max(Temp) +10, bottom = min(Temp) - 10)
        plt.legend()
        plt.title('Current Target: %f \t Current Reward: %f' %(self.env.unwrapped.engine.get_state()[0][2], r))

        plt.subplot(212)
        plt.cla()
        plt.plot(Energy, Mass, 'k', label = 'True')
        plt.plot(Energy, MassMeans, 'r', label = 'Mean')
        plt.fill_between(Energy.flatten(), MassMeans - MassSdvs, MassMeans + MassSdvs, facecolor = 'r', alpha = 0.5, label = '1 sigma range')
        plt.scatter(self.env.unwrapped.engine.EnergyIn, self.env.unwrapped.engine.get_state()[0][1])
        plt.ylabel('MassFractions')
        plt.xlabel('Energy Added (kJ)')
        plt.ylim(top = max(Mass) + 0.1, bottom = min(Mass) - 0.1)
        plt.legend()
        
        #self.fig.canvas.draw() 
        self.fig.canvas.flush_events()
        #plt.pause(0.001)

root = Tk()
#root.withdraw()
MyLoop(root)

