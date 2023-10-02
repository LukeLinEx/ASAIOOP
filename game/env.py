
import numpy as np
import gym
from gym import spaces

# 0: Empty space
# 1: Lion (Player 1)
# 2: Giraffe (Player 1)
# 3: Elephant (Player 1)
# 4: Chick (Player 1)
# 5: Chicken (Player 1)
# -1: Lion (Player 2)
# -2: Giraffe (Player 2)
# -3: Elephant (Player 2)
# -4: Chick (Player 2)
# -5: Chicken (Player 2)

class AnimalShogiEnv(gym.Env):
    def __init__(self):
        super(AnimalShogiEnv, self).__init__()
        
        # Flattened representation of the 4x3 board
        self.observation_space = spaces.Box(low=-5, high=5, shape=(12,), dtype=np.int8)
        
        # Assuming each cell is a possible action for simplicity
        self.action_space = spaces.Discrete(144 + 3 * 12)
        
        # Initialize the board and other required variables
        self.board = np.zeros(12, dtype=np.int8)
        self.player1_storage = [0,0,0] # zero giraffe, zero elephant, and zero chick to begin with. [2,2,2] at most.
        self.palyer2_storage = [0,0,0]
        self._setup_board()
        self.current_player = 1

    def _setup_board(self):
        # Resetting the board to a default configuration
        # Just a basic setup, you may change it to the standard starting position
        self.board = np.array([-2,-1,-3,0,-4,0,0,4,0,3,1,2], dtype=np.int8)


    def generate_valid_actions(self):
        valid_actions = []
    
        # Check valid moves for each piece on the board
        for from_cell in range(12):
            piece = self.board[from_cell]
            
            # Skip empty squares and then check if the piece belongs to the current player
            if self.current_player*piece > 0:
                valid_destinations = self.genearte_valid_destination(piece, from_cell)
                for to_cell in valid_destinations(piece, from_cell):
                    valid_actions.append(('move', from_cell, to_cell))
        
        # Check valid drop actions for pieces in storage
        storage = self.player1_storage if self.current_player == 1 else self.player2_storage
        for piece in storage:
            for to_cell in range(12):
                if self.board[to_cell] == 0:  # can drop only on empty squares
                    valid_actions.append(('drop', piece, to_cell))

        return valid_actions
    
    def generate_valid_destinations(self, piece, from_cell):
        valid_destinations = []

        # Convert flat index to 2D
        row, col = self._flattened_to_2d_idx(from_cell)

        # Define movement for each piece
        movements = {
            1: [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)],  # Lion
            2: [(0, 1), (0, -1), (1, 0), (-1, 0)],  # Giraffe
            3: [(1, 1), (-1, 1), (1, -1), (-1, -1)],  # Elephant
            4: [(1, 0)],  # Chick (before promotion)
            5: [(1, 0), (0, 1), (0, -1), (-1, 0)],  # Chicken (after promotion)
        }

        # Iterate for the possible movements
        for move in movements[abs(piece)]:
            new_row = row + (move[0] * self.current_player)  # Current player will determine the forward direction
            new_col = col + move[1]
            
            # Check if move is within board and not occupied by a same-player piece
            if 0 <= new_row < 4 and 0 <= new_col < 3:
                to_cell = self._2d_to_flattened_idx(new_row, new_col)
                destination_piece = self.board[to_cell]
                if self.current_player * destination_piece <= 0:  # Either empty or an opponent's piece
                    valid_destinations.append(to_cell)

        return valid_destinations


    def reset(self):
        self._setup_board()
        return self.board

    def step(self, action):
        # Here, you'd implement the game mechanics for moving the selected piece to the target square.
        # For simplicity, I'm just returning the board state as is without any action logic.
        # You'd need to implement rules for each piece, possible promotions, captures, etc.
        reward = 0
        done = False

        # Example: Check for win condition if a lion is captured
        if 1 not in self.board:
            reward = -1
            done = True
        elif -1 not in self.board:
            reward = 1
            done = True
        
        return self.board, reward, done, {}

    def render(self, mode='human'):
        # Printing the board for visualization
        print(self.board.reshape(4, 3))

    def close(self):
        pass
    
    @staticmethod
    def _flattened_to_2d_idx(flat):
        row = flat // 3
        col = flat % 3
        return row, col
    
    @staticmethod
    def _2d_to_flattened_idx(row, col):
        return row * 3 + col



if __name__ == "__main__":
    env = AnimalShogiEnv()
    # env.current_player = -1
    env.render()


    # TODO: validate a few cases, save them as test cases
    print(env.generate_valid_destinations(1, 10))






