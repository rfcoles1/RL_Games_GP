import numpy as np
import matplotlib.pyplot as plt
from phase_env import PhaseEnv
from phase_eng import *
E = PhaseEnv()

def print_status():
    print 'env state: ' + str([E.engine.T, E.engine.P, Gibbs(E.engine.T,E.engine.P)])
    print 'S state: ' + str(E.engine.get_state())
    print 'Phase: ' + str(E.engine.phase)
    print 'Reward: ' + str(E.get_reward())
            
