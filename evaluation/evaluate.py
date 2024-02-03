import time
from numpy.random  import randint
import pandas as pd
from asaioop.game.env import AnimalShogiEnv
from stable_baselines3 import PPO


model_path = "../old_models/ppo_player1_dbg_1000"
model1 = PPO.load(model_path)




def one_game():
    env = AnimalShogiEnv()
    obs = env.reset()
    done = False

    while not done:
        action, _ = model1.predict(obs, deterministic=True)
        # action = randint(300)
        obs, _, done, _ = env.step(action)
        if done:
            winner = 1
            break

        action = randint(300)
        obs, _, done, _ = env.step(action)
        if done:
            winner = -1

    return winner


def collect_game_result(num_tests):
    winners = []
    for game in range(num_tests):
        winners.append(one_game())

    return winners


if __name__ == "__main__":
    print(model_path)
    start = time.time()
    results = collect_game_result(20000)
    print(pd.Series(results).value_counts())
    print(time.time()-start)
