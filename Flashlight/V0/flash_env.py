import numpy as np
import gym
import gym.spaces
from flash_eng import Engine

class FlashEnv(gym.Env):
    def __init__(self):
        self.engine = Engine()
        self.done = False
    
        self.actions = {0: self.engine.move_left,
                        1: self.engine.move_right,
                        2: self.engine.move_down,
                        3: self.engine.move_up,
                        4: self.engine.no_move}

        self.action_space = gym.spaces.Discrete(5)
        self.reset()

    def reset(self):
        self.engine.reset()
        return self.get_state()

    def get_state(self):
        state = self.engine.get_local()
        return state

    def step(self, action):
        self.do_action(action)
        next_state = self.get_state()
        reward = self.get_reward()
        self.done = self.engine.is_done()
        return next_state, reward, self.done, {}
    
    def do_action(self, action):
        self.actions[action]()

    def get_reward(self):
        dist = -abs(self.engine.walker_pos[0] - self.engine.final_wall)
        if self.engine.is_done():
            return 10
        return dist 

    def render(self):
        return 0
        
