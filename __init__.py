from gym.envs.registration import register


#---------------------------------------------#


register(
    id = 'GP_Water-v0',
    entry_point = 'GP_Games.Boiling.V0.boil_env:HeatEnv',
    max_episode_steps = 100,
)


register(
    id = 'GP_Water-v1',
    entry_point = 'GP_Games.Boiling.V1.boil_env:HeatEnv',
    max_episode_steps = 100,
)
 

register(
    id = 'GP_Water-v2',
    entry_point = 'GP_Games.Boiling.V2.boil_env:HeatEnv',
    max_episode_steps = 100,
)

register(
    id = 'GP_Water-v3',
    entry_point = 'GP_Games.Boiling.V3.boil_env:HeatEnv',
    max_episode_steps = 100,
)

#---------------------------------------------#

"""
register(
    id = 'GP_Func-v0',
    entry_point = 'GP_Games.GP_Function.functions_env:FuncEnv',
    max_episode_steps = 100,
)
"""

#---------------------------------------------#

register(
    id = 'GP_Phase-v0',
    entry_point = 'GP_Games.Gibbs.V0.phase_env:PhaseEnv',
    max_episode_steps = 25,
)

