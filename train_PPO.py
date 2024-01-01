import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle


import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from asaioop.game.env import AnimalShogiEnv

# Assume AnimalShogiEnv is defined in the module 'your_module_name'
# Register the environment

god_view = AnimalShogiEnv()

        

gym.envs.register(
    id='p1_env',
    entry_point='asaioop.game.env:PlayerEnv',
    kwargs = {"god_view": god_view, "plyr":1})

gym.envs.register(
    id='p2_env',
    entry_point='asaioop.game.env:PlayerEnv',
    kwargs = {"god_view": god_view, "plyr":-1})

# Create environment
p1_env = DummyVecEnv([lambda: gym.make('p1_env')])
p2_env = DummyVecEnv([lambda: gym.make('p2_env')])

model_1 = PPO(
    "MlpPolicy", p1_env, verbose=0,
    learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5)


model_2 = PPO(
    "MlpPolicy", p1_env, verbose=0,
    learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5)

# Training parameters
num_games = 50
max_steps_per_game = 100

start = time.time()
# Training loop
study = False
mx_step = 0
track = []
for game in range(num_games+1):
    obs = god_view.reset()
    p1_env.reset()
    p2_env.reset()
    done = False
    
    action1, _ = model_1.predict(obs)
    obs, _, done, _ = god_view.step(action1)
    action2, _ = model_2.predict(obs)
    obs, _, done, _ = god_view.step(action2)
    p1_env.step([action1])


    for rnd in range(2, max_steps_per_game):
        action1, _ = model_1.predict(obs)
        obs, _, done, _ = god_view.step(action1)
        _, rwd2, _, _ = p2_env.step([action2]) # it's a bit confusing here but action2 was actually decided before action1 above!!!!
        if done:
            _, rwd1, _, _  = p1_env.step([action1])
            # print(f"\nAt game {game}")
            # print(f"player 1 won at round {rnd}")
            track.append([game, rnd])
            # print(f"player 1 got {rwd1}")
            # print(f"player 2 got {rwd2}")
            # god_view.render()
            break

        action2, _ = model_2.predict(obs)
        obs, _, done, _ = god_view.step(action2)
        _, rwd1, _, _  = p1_env.step([action1]) # it's a bit confusing here but action1 was actually decided before action2 above!!!!
        if done:
            _, rwd2, _, _ = p2_env.step([action2])
            # print(f"\nAt game {game}")
            # print(f"player 2 won at round {rnd}")
            track.append([game, rnd])
            # print(f"player 2 got {rwd2}")
            # print(f"player 1 got {rwd1}")
            # god_view.render()
            break
    
    # Update models after each game

    model_1.learn(total_timesteps=rnd)
    model_2.learn(total_timesteps=rnd)

    if game%10 == 0:
        print("At the game {}".format(game))
        time_last = time.time() - start
        print("Training has been {} seconds".format(time_last))

    if game%10 ==0:
        model_1.save(f"./models/ppo_player1_dbg_{game}")
        model_2.save(f"./models/ppo_player2_dbg_{game}")

        with open("./models/track.pkl", 'wb') as handle:
            pickle.dump(track, handle, protocol=pickle.HIGHEST_PROTOCOL)
