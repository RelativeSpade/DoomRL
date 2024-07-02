from vizdoom import *
import random
import time
import numpy as np

game = DoomGame()
game.load_config('../Vizdoom/scenarios/basic.cfg')
game.init()

actions = np.identity(3, dtype=int)

episodes = 10
for episode in range(episodes):
    # Create new game
    game.new_episode()
    # While game is not finished
    while not game.is_episode_finished():
        # Get the game state
        state = game.get_state()
        # Get the game screen (Whatever is on screen, enemy, armor, etc)
        img = state.screen_buffer
        # Get the game variables (Ammo)
        info = state.game_variables
        # Do a random action and get return reward
        reward = game.make_action(random.choice(actions))
        # Prints the reward
        print('reward:', reward)
        time.sleep(0.02)
    print('Result:', game.get_total_reward())
    time.sleep(2)
