# TODO: collect losing games to train
# TODO: implement some random steps if gap is less than a threshold

from copy import deepcopy
from asaioop.game.env import AnimalShogiEnv
from stable_baselines3 import PPO

model_path = "./models/ppo_player1_dbg_106000"
model = PPO.load(model_path)
print(model)
first_hand = input("Are you playing first hand?")
if first_hand == "y":
    first_hand = True
else:
    first_hand = False


env = AnimalShogiEnv()
obs = env.reset()

done = False

if not first_hand:
    print("Second")
    env.render()
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = env.step(action)

valid_steps = obs[-1]
while not done:
    print("\nAll available moves are as below:")
    for i in range(valid_steps):
        cur_env = deepcopy(env)
        cur_env.step(i, print_action=True)
        print("\n", i)
        cur_env.render()

    print("Current stage:")
    env.render()

    action = int(input("Input your move:"))
    obs, _, done, _ = env.step(action)
    if done:
        break

    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = env.step(action)
    valid_steps = obs[-1]

env.render()