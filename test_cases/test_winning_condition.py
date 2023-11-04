from copy import deepcopy
from asaioop.game.env import AnimalShogiEnv
from unittest import TestCase

# 1. Test if reaching bottom line works as expected
print("Test Number 1")
env = AnimalShogiEnv()
env.reset()

# print("Initialize Game:")
# env.render()

env.step(2)
# print("\nStep 1:")
# env.render()

env.step(5)
# print("\nStep 2:")
# env.render()

env.step(1)
# print("\nStep 3:")
# env.render()

env.step(1)
# print("\nStep 4:")
env.render()

_, _, done, _ = env.step(6)
print("\nStep 5:")
env.render()
print("Game should not be considered done here. Reward: {}".format(env.rwd_rnd))
t1 = TestCase()
msg = "The game should not have been over here. One possible issue is that your rule didn't let the " +\
     "opponent reacts to the current player's 'lion reaching the opposite bottom line.'"
t1.assertTrue(done is False, msg)
bottom_lion_crisis = deepcopy(env)

_, _, done, _ = env.step(0)
print("\nStep 6:")
env.render()
print("Game should be considered done here. The player 1 won.")
print("Reward: {}".format(env.rwd_rnd))
t2 = TestCase()
msg = "The game should have been over here because a lion has safely reached the opposite bottom line."
t2.assertTrue(done is True, msg)




# 2. Test if catching lion works as expected
print("\n\n\n\nTest Number 2. player 1's lion reached in the last step. player2 has to react.")
env = bottom_lion_crisis

print("from the last step:")
env.render()

_, _, done, _ = env.step(5)
print("\nStep 6:")
env.render()
print("The game is done and player 2 won!")
print("Reward: {}".format(env.rwd_rnd))
t3 = TestCase()
t3.assertTrue(done is True, msg)

t4 = TestCase()
msg = "player 2 should get a positive reward"
t4.assertTrue(env.rwd_rnd[1]>0, msg)

t5 = TestCase()
msg = "player 1 should get a negative reward"
t5.assertTrue(env.rwd_rnd[0]<0, msg)

# The code below show all valid moves from the current env
# print("\n")
# for i in range(10):
#     cur_env = deepcopy(env)
#     cur_env.step(i, print_action=True)
#     print("\n", i)
#     cur_env.render()
