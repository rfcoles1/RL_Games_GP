#RL_Games_GP
Reinforcement learning environments that use Gaussian Processes to provide the agent with an uncertainty measure. 

# Motivation

# Environments

![image](https://drive.google.com/uc?export=view&id=11-jHGJgbvK1PWUyTTePgk7y9G4UKbGYh)

Environment | Description
--- | ---
GP_Water-v0 | Basic Version. State contains [T, Terr, M, Merr] to achieve a fixed target.
GP_Water-v1 | Update on V0. Target values for T and M are added to the state. Target can be moved.
GP_Water-v2 | Observations actions are split for T and M. A budget is added. 

# Requirements 
Requirements for just the environments: numpy, GPy, matplotlib, gym
Addtional Requirements for video/plotting: pylab, Tkinter

# Author
Rory Coles 

# Acknowledgments
CLEAN group
