import numpy as np
import gym
import gym.spaces
from boil_engine import Engine

#np.random.seed(1)

#in the style of OpenAi gym
class HeatEnv(gym.Env):
    def __init__(self):

        self.engine = Engine()
        self.done = False #There is no 'done' state, game runs until time limit 

        self.incr = 10 #amount of energy added/taken per step 

        self.action_space = gym.spaces.Discrete(4)
        #actions are (in order) take energy, do nothing, add energy, add a data point 
        self.reset()
    

    def reset(self):
        #change as desired
        self.TargetE = np.random.rand()*(self.engine.maxE - self.engine.minE) + self.engine.minE
        self.TargetTemp, self.TargetMass = self.engine.get_true_value(self.TargetE)

        #self.TargetTemp = np.random.rand()*100
        #self.TargetMass = np.array([0, 1, 0])
        
        self.engine.set_target(self.TargetTemp, self.TargetMass) 

        self.engine.reset()
        return self.get_state()
        
    def get_state(self):
        state = self.engine.get_state()

        T = (state[0,0] - self.engine.minT)/(self.engine.maxT - self.engine.minT)
        Terr = state[1,0]/(self.engine.maxT - self.engine.minT)
        Temp = np.vstack([T,Terr])

        M = state[0,1]/2.0
        Merr = state[1,1]/2.0
        Mass = np.vstack([M,Merr])

        TargetT = (state[0,2] - self.engine.minT)/(self.engine.maxT - self.engine.minT)
        TargetM = (state[1,2])/2.0
        Target = np.vstack([TargetT, TargetM])

        return np.hstack([Temp, Mass, Target])

    def step(self, action):
        self.do_action(action)
        next_state = self.get_state()
        reward = self.get_reward()
        return next_state, reward, self.done, {}

    #reward scheme is under revision, need to be scaled such that one number does not dominate, i.e. temperature difference 
    def get_reward(self):
        state = self.get_state()
        true_T, true_M = self.engine.get_true_value(self.engine.EnergyIn)
        
        true_T = (true_T - self.engine.minT)/(self.engine.maxT - self.engine.minT)

        dT_pred = abs(state[0][0] - state[0][2])
        dT_true = abs(true_T - state[0][2])
        Terr = state[1][0]
        T_reward = -(dT_pred + dT_true + Terr)

        #decode the mass to get the values for each phase
        #means that only being in the right phase affects reward 
        targetmass = self.engine.decodeMass(state[1][2]*2)
        
        dM_pred = abs(self.engine.decodeMass(state[0][1]*2) - targetmass)
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
