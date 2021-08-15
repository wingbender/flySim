from gym.envs.registration import register

register(
    id='flySim-v1',
    entry_point='gym_flySim.envs:flySimEnv',
)