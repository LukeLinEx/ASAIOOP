import numpy as np
import random
from gym import spaces
from asaioop.game.env import AnimalShogiEnv


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, n_actions=None):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions  # May not be needed if we use generate_valid_actions

    def select_action(self, state, valid_actions): # TODO <<<<<< Not good
        state = tuple(state)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = self.q_table.get(state, {})
            # Filter available actions by their Q values or select a random action if Q values not present
            actions = [action for action in valid_actions if action in q_values.keys()]
            if actions:
                return max(actions, key=lambda action: q_values.get(action, 0))
            else:
                return random.choice(valid_actions)

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
    # Q learning from scratch
    import time 

    start = time.time()
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

    print(time.time()-start)
