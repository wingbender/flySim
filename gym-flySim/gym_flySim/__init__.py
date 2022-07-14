from gym.envs.registration import register

register(
    id='flySim-v0',
    entry_point='gym_flySim.envs:flySimEnv',
    max_episode_steps=1000,
)

register(
    id='flySim1D-v0',
    entry_point='gym_flySim.envs:flySimEnv_1D',
    max_episode_steps=1000,
)