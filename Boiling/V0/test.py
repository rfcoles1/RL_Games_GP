import numpy as np
import matplotlib.pyplot as plt
from boil_env import HeatEnv
E = HeatEnv()

def print_status():
    print('env state: %.2f, %.2f ' % (E.engine.T , E.engine.M))
    print 'Eng state: ' + str(E.engine.get_state())
    print 'Net state: ' + str(E.get_state())
    print('Reward: %.2f' % E.get_reward())

def move_right():
    E.step(2)
    E.step(3)
    print_status()

