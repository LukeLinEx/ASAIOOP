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
