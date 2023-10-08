import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from asaioop.game.env import AnimalShogiEnv



def encode_action(action_type, arg1, arg2):
    """
    Encodes an action into a single integer.
    
    Parameters:
        action_type (str): 'move' or 'drop'
        arg1 (int): if 'move' -> from_cell, if 'drop' -> piece_type
        arg2 (int): if 'move' -> to_cell, if 'drop' -> to_cell
        
    Returns:
        int: Encoded action.
    """
    assert action_type in ['move', 'drop']
    
    if action_type == 'move':
        # Ensure that the cell numbers are valid
        assert 0 <= arg1 < 12
        assert 0 <= arg2 < 12
        
        return arg1 * 12 + arg2
    
    elif action_type == 'drop':
        # Ensure that the piece_type and cell number are valid
        assert 2 <= arg1 <= 4  # Assuming piece types are 2, 3, 4
        assert 0 <= arg2 < 12
        
        return 144 + (arg1 - 2) * 12 + arg2



# Assume AnimalShogiEnv is defined in the module 'your_module_name'
# Register the environment
gym.envs.register(
    id='AnimalShogi-v0',
    entry_point='asaioop.game.env:AnimalShogiEnv',
)

test_player =-1


# Create environment
env = DummyVecEnv([lambda: gym.make('AnimalShogi-v0')])  # Single environment instance

obs  = env.reset()
env.envs[0].env.render()

if test_player ==1:
    player_model = PPO.load("ppo_player1")
else:
    player_model = PPO.load("ppo_player2")

obs = obs.astype(np.float32)  # Ensure the numpy array is float32
obs_tensor = torch.as_tensor(obs, device=player_model.device).unsqueeze(0)



mx = -np.inf
mn = np.inf
step = 181
valid_actions = []
invalid_actions = []

if test_player!=1:
    act_ = 3
    action = np.array([act_], np.int8)
    _ = env.step(action)
print("Player:", env.get_attr('current_player')[0])


for i in range(180):
    x,y,z = env.envs[0].env.unwrapped.decode_action(i)
    act_ = (x, np.array([y], np.int8), np.array(z, np.int8))

    action = np.array([i])
    action_tensor = torch.as_tensor(action, device=player_model.device).unsqueeze(0)
    prob_tensor = player_model.policy.evaluate_actions(obs_tensor, action_tensor)[1]
    prob = prob_tensor.detach().numpy()[0,0]

    if act_ in env.envs[0].env.unwrapped.generate_valid_actions():
        valid_actions.append( (i, prob))
    else:
        invalid_actions.append((i, prob))

for i, prob in valid_actions:
    print(prob)
    print(env.envs[0].env.unwrapped.decode_action(i))
print("\n")
for i, prob in invalid_actions[:10]:
    print(prob)
    print(env.envs[0].env.unwrapped.decode_action(i))


# action_probs = player1_model.action_probability(obs)
# asenv=env.envs[0].env
# asenv.render()
# print(asenv.decode_action(action))
# obs = env.envs[0].env.unwrapped
# print(obs.generate_valid_actions())
# print([encode_action(*act) for act in   obs.generate_valid_actions()])
# print(obs.step(3))
# env.envs[0].env.render()
