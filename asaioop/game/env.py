
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
    def __init__(self, init_board=None):
        super(AnimalShogiEnv, self).__init__()
        
        # Flattened representation of the 4x3 board
        self.observation_space = spaces.Box(low=-5, high=5, shape=(12,), dtype=np.int8)
        
        # Assuming each cell is a possible action for simplicity
        self.action_space = spaces.Discrete(144 + 3 * 12)
        
        # Initialize the board and other required variables
        self.board = np.zeros(12, dtype=np.int8)
        self.player1_storage = [0,0,0] # zero giraffe, zero elephant, and zero chick to begin with. [2,2,2] at most.
        self.player2_storage = [0,0,0]
        self._setup_board(init_board)
        self.current_player = 1

    def _setup_board(self, init_board=None):
        # Resetting the board to a default configuration
        # Just a basic setup, you may change it to the standard starting position
        if init_board is not None:
            self.board = np.array(init_board, dtype=np.int8)
        else:
            self.board = np.array([2,1,3,0,4,0,0,-4,0,-3,-1,-2], dtype=np.int8)


    def generate_valid_actions(self):
        valid_actions = []
    
        # Check valid moves for each piece on the board
        for from_cell in range(12):
            piece = self.board[from_cell]
            
            # Skip empty squares and then check if the piece belongs to the current player
            if self.current_player*piece > 0:
                valid_destinations = self.generate_valid_destinations(piece, from_cell)
                for to_cell in valid_destinations:
                    valid_actions.append(('move', from_cell, to_cell))
        
        # Check valid drop actions for pieces in storage
        storage = self.player1_storage if self.current_player == 1 else self.player2_storage
        for i in range(len(storage)):
            piece = i + 2
            for to_cell in range(12):
                if storage[i] > 0 and self.board[to_cell] == 0:  # can drop only on empty squares
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
        """
        action: a tuple (action_type, from_location, to_location)
            - action_type: "move" or "drop"
            - from_location: index in the flattened board or piece type (when dropping a piece)
            - to_location: index in the flattened board where the piece should move/drop
        """
        reward = 0
        done = False
        info = {}
        
        action_type, tbd, to_location = action
        
        if action_type == "move":
            from_location = tbd
            # Perform the move
            piece = self.board[from_location]
            target = self.board[to_location]
            
            # Check for capture
            if target != 0:
                # Add the piece to the storage, converting it to a positive value if it is negative
                self.add_to_storage(abs(target))
                
            # Update the board
            self.board[from_location] = 0
            self.board[to_location] = piece
            
            # Check for promotion (assuming chick (4) promotes to chicken (5) and it promotes only when it moves into the final rank)
            if piece == 4 and 9 <= to_location <= 11:  # Moving to the last rank
                self.board[to_location] = 5  # Promote chick to chicken
            elif piece == -4 and 0 <= to_location <= 2:  # Moving to the last rank
                self.board[to_location] = -5  # Promote chick to chicken
        
        elif action_type == "drop":
            # Place the piece from the storage to the board
            piece = abs(tbd) # here we make piece positive as we will use it to get the index in storage.

            if self.remove_from_storage(piece): 
                self.board[to_location] = self.current_player*piece
            else:
                # Handle invalid action, like trying to drop a piece not in storage
                # Depending on your approach you may choose to penalize the model here
                pass
        
        # Check for win conditions
        # 1. If a lion is captured
        if 1 not in self.board:
            reward = 1  # Player 2 wins
            done = True
        elif -1 not in self.board:
            reward = 1  # Player 1 wins
            done = True
            
        # 2. If a lion reaches the opponent's bottom row
        if 1 in self.board[9:12]:
            reward = 1  # Player 1 wins
            done = True
        # Player 2's lion reaches opponent's bottom row
        elif -1 in self.board[0:3]:
            reward = 1  # Player 2 wins
            done = True
            
        self.current_player *= -1


        return self.board.copy(), reward, done, info
    
    def add_to_storage(self, piece):
        """Adds a piece to the player's storage."""
        # Assuming piece is always positive
        storage = self.player1_storage if self.current_player==1 else self.player2_storage
        
        if piece == 1:  # Lions cannot be captured
            return False
        
        # Map other piece types to storage index
        piece_type_to_storage_idx = {2: 0, 3: 1, 4: 2, 5: 2}
        storage_idx = piece_type_to_storage_idx.get(abs(piece), None)
        
        # Update storage
        if storage_idx is not None:
            storage[storage_idx] += 1
            return True
        else:
            return False
        
    def remove_from_storage(self, piece):
        """Removes a piece from the player's storage."""
        # Assuming piece is always positive
        storage = self.player1_storage if self.current_player==1 else self.player2_storage
        
        piece_type_to_storage_idx = {2: 0, 3: 1, 4: 2, 5: 2}
        storage_idx = piece_type_to_storage_idx.get(abs(piece), None)
        
        # Update storage
        if storage_idx is not None and storage[storage_idx] > 0:
            storage[storage_idx] -= 1
            return True
        else:
            return False

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


# TODO: generating valid actions on the fly doesn't seem to fit stable baseline's structure. I will have to rewrite the step method to take integer action
#       and then decode it to something executable in the context of ani shogi. With this, the step function will have to penalize greatly on an invalid move
#       and shut the game.

