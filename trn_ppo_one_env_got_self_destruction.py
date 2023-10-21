import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np


import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Assume AnimalShogiEnv is defined in the module 'your_module_name'
# Register the environment
gym.envs.register(
    id='AnimalShogi-v0',
    entry_point='asaioop.game.self_destruction_env:AnimalShogiEnv',
)

# Create environment
env = DummyVecEnv([lambda: gym.make('AnimalShogi-v0')])  # Single environment instance




class MyPPO(PPO):
    def __init__(self, policy, env, **kwargs):
        super(MyPPO, self).__init__(policy, env, verbose=kwargs["verbose"])

    def predict(self, observation, state=None, mask=None, deterministic=False):
        num_available_actions =observation[0][-1]
        action, other = super(MyPPO, self).predict(observation, state, mask, deterministic)
        action = action % num_available_actions

        return action, other

model_1 = MyPPO("MlpPolicy", env, verbose=0, clip_range=0.3)
model_2 = MyPPO("MlpPolicy", env, verbose=0, clip_range=0.3)
# model_1 = PPO("MlpPolicy", env, verbose=0)
# model_2 = PPO("MlpPolicy", env, verbose=0)

# Training parameters
num_games = 300
max_steps_per_game = 100

start = time.time()
# Training loop
study = False
nontrivial = []
strategic = set()
mx_step = 0
for game in range(num_games):
    obs = env.reset()

    done = False
    env_ = env.envs[0].env

    game_steps = []
    for step in range(max_steps_per_game):
        # Decide which model to use based on the current player
        cp = env.get_attr('current_player')[0]
        # print("current player:", cp)
        model = model_1 if cp == 1 else model_2

        # Agent takes action
        action, _ = model.predict(obs)

        obs, reward, done, _ = env.step(action)
        # print(obs[0][:12].reshape(4,3))
        # print(reward, done)
        
        # game_steps.append((obs, reward, cp))
        

        # If the game is done, break
        done = done[0]
        if step > mx_step:
            mx_step = step
        if done:
            # if step < 3:
            #     print("Oh~~")
            #     nontrivial.append(game)
            #     output = []
            #     for obs_ in game_steps:
            #         board = (obs_[0][0][:12]).reshape(4,3)
            #         storage1 = obs_[0][0][12:15]
            #         storage2 = obs_[0][0][15:18]
            #         num_curr_actions = obs_[0][0][-1]
            #         rwd = obs_[1]
            #         cp_ = obs_[2]

            #         to_show = [step, cp_, rwd, board, storage1, storage2, num_curr_actions]
            #         to_show = "\n".join([str(x) for x in to_show])
            #         output.append(to_show)
            #         if rwd > 0:
            #             strategic.add(game)
            #             study = True
            break
        # if done:
        #     if step > 1:
        #         print(step, obs)
        #         env.envs[step].env.render()
        #         print(cp, game, step, reward)
        #     break
    
    # Update models after each game
    st = time.time()
    if cp == 1:  # Last action was taken by player 1
        model_1.learn(total_timesteps=step)
        model_2.learn(total_timesteps=step-1)
    else:  # Last action was taken by player 2
        model_1.learn(total_timesteps=step-1)
        model_2.learn(total_timesteps=step)
    print(time.time()-st, game, step)

    # if game % 50==0:
    #     print("\nGame:", game, mx_step)
    #     print(time.time()-start)
    #     mx_step = 0

    # if study:
    #     for s in output:
    #         print(s)

    #     print(time.time()-start)
    #     print(len(nontrivial))
    #     print(nontrivial)
        


    ###### >>>>>>>>>>>>>>> Read policy after training step
    # prob_tensor = model_1.policy.evaluate_actions(obs_tensor, action_tensor)[1]
    # prob = prob_tensor.detach().numpy()[0]
    # print("after:", prob)
    ###### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

model_1.save("ppo_player1_dbg")
model_2.save("ppo_player2_dbg")

if not study:
    print(time.time()-start)
    print(len(nontrivial))

