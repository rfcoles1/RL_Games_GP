import numpy as np 
import gym
import gym.spaces

from phase_eng import Engine
 
class PhaseEnv(gym.Env):
    def __init__(self):
        self.engine = Engine()
        self.done = False

        self.TargetG = -15
        self.TargetPhase = 'i'

        self.action_space = gym.spaces.Discrete(6)
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

    def do_action(self, action):
        if action == 0:
            self.engine.apply_heat(-1)
        if action == 1:
            self.engine.apply_heat(1)
        if action == 2:
            self.engine.apply_work(-1)
        if action == 3:
            self.engine.apply_work(1)
        #action 4 reserved as a null action 
        if action == 5:
            self.engine.add_data()
  
    """ 
    def get_reward(self):
        curr_phase = self.engine.phase
        if curr_phase == self.TargetPhase:
            return 0
        else:
            return -1 
    """

    def get_reward(self):
        state = self.get_state()
        trueG = self.engine.get_true(self.engine.T,self.engine.P)[0]
        msd_fake = abs(state[0] - self.TargetG)
        msd_true = abs(trueG - self.TargetG)
        return -msd_fake - msd_true - state[1]

    

    def render(self, mode = 'human'):
        return 0

    
