import numpy as np
import random
import gym
from gym import spaces



########### This whole thing is now in env.py ###################################
# class AnimalShogiEnv(gym.Env):
#     def __init__(self):
#         super(AnimalShogiEnv, self).__init__()
        
#         # Flattened representation of the 4x3 board
#         self.observation_space = spaces.Box(low=0, high=..., shape=(12,), dtype=np.float32)
        
#         # Assuming 'n' possible actions
#         # self.action_space = spaces.Discrete(144 + 3 * 12)
        
#         # Initialize the board and other required variables
#         # ...

#     def reset(self):
#         # Reset the board and return the initial state
#         return ...

#     def step(self, action):
#         # Execute the action, return the new state, reward, done, and any additional info
#         # ...
#         return next_state, reward, done, {}

#     def render(self, mode='human'):
#         # Optional: Implement rendering for visualization
#         pass

#     def close(self):
#         # Clean up the environment
#         pass
########### ^^^^^^^^^^^^^^^^^^^^^^^^   ############################################



class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, n_actions=None):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def select_action(self, state):
        state = tuple(state)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.n_actions))
        else:
            return max(self.q_table.get(state, {}).keys(), key=lambda action: self.q_table[state].get(action, 0))

    def learn(self, state, action, reward, next_state):
        state = tuple(state)
        next_state = tuple(next_state)
        best_next_q_value = max(self.q_table.get(next_state, {}).values(), default=0)
        current_q_value = self.q_table.get(state, {}).get(action, 0)
        new_q_value = current_q_value + self.alpha * (reward + self.gamma * best_next_q_value - current_q_value)
        
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q_value

if __name__ == "__main__":
    env = AnimalShogiEnv()
    n_actions = env.action_space.n

    player1 = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1, n_actions=n_actions)
    player2 = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1, n_actions=n_actions)

    total_timesteps = 1000000
    for timestep in range(total_timesteps):
        state = env.reset()
        done = False

        while not done:
            action1 = player1.select_action(state)
            next_state, reward1, done, info = env.step(action1)
            
            if not done:
                action2 = player2.select_action(next_state)
                next_state, reward2, done, info = env.step(action2)
                player1.learn(state, action1, reward1, next_state)
                state = next_state
                player2.learn(state, action2, reward2, next_state)
            else:
                player1.learn(state, action1, reward1, next_state)
