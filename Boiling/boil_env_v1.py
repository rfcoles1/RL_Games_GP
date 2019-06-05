import numpy as np
import gym
import gym.spaces
from boil_engine_v1 import Engine

#in the style of OpenAi gym
class HeatEnv(gym.Env):
    def __init__(self):
        self.engine = Engine()
        self.done = False #There is no 'done' state, game runs until time limit 

        #change as desired
        self.TargetTemp = 80
        self.TargetMass = np.array([0, 1, 0])
        self.engine.set_target(self.TargetTemp, self.TargetMass) 

        self.incr = 10 #amount of energy added/taken per step 

        self.action_space = gym.spaces.Discrete(4)
        #actions are (in order) take energy, do nothing, add energy, add a data point 
        self.reset()

    def reset(self):
        self.engine.reset()
        return self.engine.get_state()

    def get_state(self):
        return self.engine.get_state()

    def step(self, action):
        self.do_action(action)
        next_state = self.engine.get_state()
        reward = self.get_reward()
        return next_state, reward, self.done, {}

    #reward scheme is under revision, need to be scaled such that one number does not dominate, i.e. temperature difference 
    def get_reward(self):
        state = self.get_state()
        true_T, true_M = self.engine.get_true_value(self.engine.EnergyIn)
        
        dT_pred = abs(state[0][0] - state[0][2])
        dT_true = abs(true_T - state[0][2])
        Terr = state[1][0]
        T_reward = -(dT_pred + dT_true + Terr)
        

        targetmass = self.engine.decodeMass(state[1][2])
        dM_pred = abs(self.engine.decodeMass(state[0][1]) - targetmass)
        dM_true = abs(true_M - targetmass)
        Merr = state[1][1]
        M_reward = -(np.sum(dM_true) + np.sum(dM_pred) + Merr)
        
        return T_reward + M_reward

    def do_action(self, action):
        if action == 3:
            self.engine.add_data()
        else:
            self.engine.EnergyIn += (action-1)*self.incr
        Tpred, Mpred = self.engine.get_pred(np.array([self.engine.EnergyIn]).reshape(-1,1))
        self.engine.T = Tpred[0].flatten()[0]
        self.engine.Terr = Tpred[1].flatten()[0]
        self.engine.M = Mpred[0].flatten()[0]
        self.engine.Merr = Mpred[1].flatten()[0]

    def render(self, mode = 'human'):
        #TODO 
        return 0
