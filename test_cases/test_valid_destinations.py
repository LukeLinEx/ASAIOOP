import numpy as np
from asaioop.game.env import *

def string_to_2d_array(s):
    # Remove unwanted characters and split the string into lines
    lines = s.replace("[", "").replace("]", "").split("\n")
    
    # Create a list of lists with the integer values
    array_2d = [[int(num) for num in line.split()] for line in lines if line.strip()]
    
    # Convert the list of lists into a 2D NumPy array
    return np.array(array_2d)


def viz_test_valid_destination_case(s, location, current_player=1):
    env = AnimalShogiEnv()
    flat = string_to_2d_array(s).reshape(-1,)
    for i in range(len(flat)):
        env.board[i] = flat[i]
    env.current_player = current_player

    piece = env.board[location]
    print("\n")
    env.render()
    if piece*current_player<0:
        print(f"Player:{current_player}, Location: {location}, Piece: {piece}\nError: The piece doesn't belong to the player.")
    else:
        print(f"Player:{current_player}, Location: {location}, Piece: {piece}\nValid destination: {env.generate_valid_destinations(piece, location)}")


## player 1
s = """[[2  1  3]
 [ 0  4  0]
 [ 0 -4  0]
 [-3 -1 -2]]"""
location = 0
current_player = 1
viz_test_valid_destination_case(s, location, current_player)


s = """[[2  1  3]
 [ 0  4  0]
 [ 0 -4  0]
 [-3 -1 -2]]"""
location = 1
current_player = 1
viz_test_valid_destination_case(s, location, current_player)

s = """[[2  1  3]
 [ 0  4  0]
 [ 0 -4  0]
 [-3 -1 -2]]"""
location = 2
current_player = 1
viz_test_valid_destination_case(s, location, current_player)


s = """[[2  1  3]
 [ 0  4  0]
 [ 0 -4  0]
 [-3 -1 -2]]"""
location = 4
current_player = 1
viz_test_valid_destination_case(s, location, current_player)


s = """[[2  1  3]
 [ 0  4  0]
 [ 0 -4  0]
 [-3 -1 -2]]"""
location = 7
current_player = 1
viz_test_valid_destination_case(s, location, current_player)


### Player 2
s = """[[2  1  3]
 [ 0  4  0]
 [ 0 -4  0]
 [-3 -1 -2]]"""
location = 9
current_player = -1
viz_test_valid_destination_case(s, location, current_player)

s = """[[2  1  3]
 [ 0  4  0]
 [ 0 -4  0]
 [-3 -1 -2]]"""
location = 10
current_player = -1
viz_test_valid_destination_case(s, location, current_player)

s = """[[2  1  3]
 [ 0  4  0]
 [ 0 -4  0]
 [-3 -1 -2]]"""
location = 11
current_player = -1
viz_test_valid_destination_case(s, location, current_player)


s = """[[2  1  3]
 [ 0  4  0]
 [ 0 -4  0]
 [-3 -1 -2]]"""
location = 7
current_player = -1
viz_test_valid_destination_case(s, location, current_player)


s = """[[2  1  3]
 [ 0  4  0]
 [ 0 -4  0]
 [-3 -1 -2]]"""
location = 4
current_player = -1
viz_test_valid_destination_case(s, location, current_player)


def test_generate_valid_destinations():
    print("\nChatGPT Tests Result:")
    env = AnimalShogiEnv()

    # Test Lion movement for Player 1 (Piece 1)
    env.board = np.zeros(12, dtype=np.int8)
    env.board[4] = 1  # Lion at the center
    assert set(env.generate_valid_destinations(1, 4)) == {0,1,2,3,5,6,7,8}

    # Test Giraffe movement for Player 1 (Piece 2)
    env.board = np.zeros(12, dtype=np.int8)
    env.board[4] = 2  # Giraffe at the center
    assert set(env.generate_valid_destinations(2, 4)) == {1, 3, 5, 7}

    # Test Elephant movement for Player 1 (Piece 3)
    env.board = np.zeros(12, dtype=np.int8)
    env.board[4] = 3  # Elephant at the center
    assert set(env.generate_valid_destinations(3, 4)) == {0, 2, 6, 8}

    # Test Chick movement for Player 1 (Piece 4)
    env.board = np.zeros(12, dtype=np.int8)
    env.board[4] = 4  # Chick at the center
    assert set(env.generate_valid_destinations(4, 4)) == {7}

    # Test Chicken movement for Player 1 (Piece 5)
    env.board = np.zeros(12, dtype=np.int8)
    env.board[4] = 5  # Chicken at the center
    assert set(env.generate_valid_destinations(5, 4)) == {1, 3, 5, 7}

    # Test Lion movement for Player 2 (Piece -1)
    env.board = np.zeros(12, dtype=np.int8)
    env.board[4] = -1  # Lion at the center
    env.current_player = -1  # Switching to Player 2
    assert set(env.generate_valid_destinations(-1, 4)) == {1, 3, 5, 0, 2, 6, 7, 8}

    # Test Chick movement for Player 2 (Piece -1)
    env.board = np.zeros(12, dtype=np.int8)
    env.board[4] = -4  # Lion at the center
    env.current_player = -1  # Switching to Player 2
    assert set(env.generate_valid_destinations(-4, 4)) == {1}

    print("All tests passed!")


test_generate_valid_destinations()
s = """[[0  0  0]
[ 0  3  0]
[ 0  0  0]
[-3 -1 -2]]"""
location = 4
current_player = 1
viz_test_valid_destination_case(s, location, current_player)


print(string_to_2d_array("""[[ 2  1  3]
 [ 0  4  0]
 [ 0 -4  0]
 [-3 -1 -2]]"""))

from asaioop.game.env import AnimalShogiEnv
env = AnimalShogiEnv()

# env._flattened_to_2d_idx(flat)
#         # row = flat // 3
#         # col = flat % 3
#         # return row, col
    
print(env._2d_to_flattened_idx(0,0))