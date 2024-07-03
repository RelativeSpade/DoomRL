import cv2
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from vizdoom import *


class VizDoomGym(Env):

    # Function that is called when we start the Env
    def __init__(self, render=False, configMap='../Vizdoom/scenarios/basic.cfg'):
        super().__init__()
        # Set up the game
        self.game = DoomGame()
        # Choose the level
        self.game.load_config(configMap)

        # If you are rendering the game
        if render:
            self.game.set_window_visible(True)
        else:
            self.game.set_window_visible(False)

        # Start the game
        self.game.init()

        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(7)

        # HEALTH DAMAGE_TAKEN DAMAGECOUNT AMMO
        self.damage_taken = 0
        self.damage_count = 0
        self.ammo = 52

    # Function that is called on every Ai action (or step)
    def step(self, action):
        # Specify actions and take step
        actions = np.identity(7, dtype=np.uint8)
        movement_reward = self.game.make_action(actions[action], 4)
        reward = 0
        damage_taken_delta = 0
        damage_count_delta = 0
        ammo_delta = 0
        debug = False
        # Get necessary returns from client
        if self.game.get_state():
            image = self.game.get_state().screen_buffer
            image = self.grayscale(image)

            # Reward shaping
            game_variables = self.game.get_state().game_variables
            health, damage_taken, damage_count, ammo = game_variables

            # Calculate reward deltas (delta means difference)
            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken
            damage_count_delta = damage_count - self.damage_count
            self.damage_count = damage_count
            ammo_delta = ammo - self.ammo
            self.ammo = ammo

            reward = (movement_reward * 0.75 +
                      damage_taken_delta * 10 +
                      damage_count_delta * 50 +
                      ammo_delta * 5)

            info = ammo
        else:
            image = np.zeros(self.observation_space.shape)
            info = 0

        info = {"info": info}

        if debug:
            print("Action: {} Reward: {} \nMovement: {} Ammo: {} \nDamage Dealt: {} Damage Taken: {}".format(
                actions[action], reward,
                movement_reward, ammo_delta,
                damage_count_delta, damage_taken_delta))

        done = self.game.is_episode_finished()
        return image, reward, done, info

    # Function that is called to close the game
    def close(self):
        self.game.close()

    # Define how to render the game or environment
    def render(self, **kwargs):
        pass

    # Function to grayscale the game frame and resize it
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))
        return state

    # What happens when we start a new game
    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        self.damage_taken = 0
        self.damage_count = 0
        self.ammo = 52
        return self.grayscale(state)
