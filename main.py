from asaioop.game.env import AnimalShogiEnv


if __name__ == "__main__":
    # With the current setting, player 1 should be on the top of the board,
    # player 2 on the bottom.


    # Do you want to test only valid actions?
    only_valid = True
    custom_init = True



    import random
    specific_board = [1,0,0,5,5,5,0,0,0,0,0,-1]
    # Initialize environment
    if custom_init:
        env = AnimalShogiEnv(specific_board)
    else:
        env = AnimalShogiEnv()

    # Reset environment to start state
    state = env.reset()

    done = False
    env.render()

    while not done:
        # Generate valid actions for the current player
        valid_actions = env.generate_valid_actions()
        print(valid_actions)
        # Select a random action from the list of valid actions
        action = 3#random.choice(range(180))

        if only_valid:
            while env.decode_action(action) not in valid_actions:
                action = random.choice(range(180))
        
        # Apply the selected action and get the new state and reward
        print(env.current_player, action, env.decode_action(action))
        next_state, reward, done, info = env.step(action)
        
        # Render the current state of the environment
        env.render()
        print(env.player1_storage)
        print(env.player2_storage)
        
        # Display reward and done status
        print(f"Reward: {reward}, Done: {done}")
        
        # Check if game is still ongoing
        if not done:
            # Prompt user if they want to continue
            user_input = input("\n\nStop? (Y to stop, any other key to continue): ")
            
            # Break the loop if user input is "Y" or "y"
            if user_input.lower() == "y":
                break
        else:
            # Announce the winner
            print(f"Player {2 if env.current_player == 1 else 1} wins!")

    # Close the environment
    env.close()

    # test if player2 drop a piece correctly
    # env = AnimalShogiEnv(specific_baord)
    # env.player2_storage = [0,0,1]
    # env.step(  ("move", 0, 1)  )
    # env.step(  ("drop", 4, 4)  )
    # env.render()

