# RL_Games_GP
Reinforcement learning environments that use Gaussian Processes to provide the agent with an uncertainty measure. 

# Motivation
In the scenarios where an RL agent does not know the state with full certainty, it still must be able to make informed decisions.
We have designed a series of RL environments in which a model of the environment is made as the agent plays the game.
This allows for the agent to have a measure of uncertainty between discrete observations which provide the ground truth.

# Environments
In the GP_Water environments, the agent must heat/cool the a fixed amount of water to a certain point.
To do so, it must observe the Temperature and Mass Fraction of the solid/liquid/gas phases and either add or remove energy.

![image](https://drive.google.com/uc?export=view&id=11-jHGJgbvK1PWUyTTePgk7y9G4UKbGYh)

Environment | Description
--- | ---
GP_Water-v0 | Basic Version. State contains [T, Terr, M, Merr] to achieve a fixed target.
GP_Water-v1 | Target values for T and M are added to the state. Target can be moved.
GP_Water-v2 | Observations actions are split for T and M. Each observation now comes with a cost. 
GP_Water-v3 | Energy added to action, can remove either T or M from state and reach a target.

# Requirements 
Requirements for the environments: numpy, GPy, matplotlib, gym

Addtional Requirements for interactive demo: Tkinter

# Author
Rory Coles 

# Acknowledgments
CLEAN group
