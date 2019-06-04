from gym.envs.registration import register


#---------------------------------------------#

register(
    id = 'GP_Water-v0',
    entry_point = 'GP_Games.Boiling.boil_env_v0:HeatEnv',
    max_episode_steps = 100,
)

register(
    id = 'GP_Water-v1',
    entry_point = 'GP_Games.Boiling.boil_env_v1:HeatEnv',
    max_episode_steps = 100,
)

#---------------------------------------------#

register(
    id = 'GP_Func-v0',
    entry_point = 'GP_Games.GP_Function.functions_env:FuncEnv',
    max_episode_steps = 100,
)

#---------------------------------------------#

register(
    id = 'GP_Phase-v0',
    entry_point = 'GP_Games.Gibbs.phase_env:PhaseEnv',
    max_episode_steps = 25,
)

