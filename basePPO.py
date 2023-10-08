import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np


import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from asaioop.game.env import AnimalShogiEnv

# Assume AnimalShogiEnv is defined in the module 'your_module_name'
# Register the environment
gym.envs.register(
    id='AnimalShogi-v0',
    entry_point='asaioop.game.env:AnimalShogiEnv',
)

# Create environment
env = DummyVecEnv([lambda: gym.make('AnimalShogi-v0')])  # Single environment instance

model_1 = PPO("MlpPolicy", env, verbose=0)
model_2 = PPO("MlpPolicy", env, verbose=0)

# Training parameters
num_games = 3000
max_steps_per_game = 20

start = time.time()
# Training loop
for game in range(num_games):
    obs = env.reset()
    # obs_tensor = torch.as_tensor(obs, device=model_1.device).unsqueeze(0)


    done = False
    env_ = env.envs[0].env
    
    ######## >>>>>>>>>>>>>>>>>>>>>>>>> Test a particular action
    # act_ = 3
    # action_tensor = torch.as_tensor(act_, device=model_1.device).unsqueeze(0)
    # action = np.array([act_], np.int8)
    # prob_tensor = model_1.policy.evaluate_actions(obs_tensor, action_tensor)[1]
    # prob = prob_tensor.detach().numpy()[0]
    # print("before:", prob)
    #################   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



    for step in range(max_steps_per_game):
        # Decide which model to use based on the current player
        cp = env.get_attr('current_player')[0]
        # print("current player:", cp)
        model = model_1 if cp == 1 else model_2
        
        # Agent takes action
        action, _ = model.predict(obs)
        
        ###### >>>>>>>>>>>>>>>>>> Inspect actions
        # decoded_action = env_.decode_action(action)
        # print(action, decoded_action)
        # print("Valid actions:", env_.generate_valid_actions())
        # print("valid action:", decoded_action in env_.generate_valid_actions())
        ###### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        obs, reward, done, _ = env.step(action)
        # print(reward, done)
        
        # If the game is done, break
        done = done[0]
        if done:
            if step > 3:
                print(game, step, "Game over")
            break
    
    # Update models after each game
    if cp == 1:  # Last action was taken by player 1
        model_1.learn(total_timesteps=step)
        model_2.learn(total_timesteps=step-1)
    else:  # Last action was taken by player 2
        model_1.learn(total_timesteps=step-1)
        model_2.learn(total_timesteps=step)

    if game % 100==0:
        print("Game:", game, reward)

    ###### >>>>>>>>>>>>>>> Read policy after training step
    # prob_tensor = model_1.policy.evaluate_actions(obs_tensor, action_tensor)[1]
    # prob = prob_tensor.detach().numpy()[0]
    # print("after:", prob)
    ###### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

model_1.save("ppo_player1")
model_2.save("ppo_player2")
print(time.time()-start)
