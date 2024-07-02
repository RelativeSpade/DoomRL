import cv2
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from vizdoom import *


class VizDoomGym(Env):

    # Function that is called when we start the Env
    def __init__(self, render=False):
        super().__init__()
        # Set up the game
        self.game = DoomGame()
        # Choose the level
        self.game.load_config('../Vizdoom/scenarios/basic.cfg')

        # If you are rendering the game
        if render:
            self.game.set_window_visible(True)
        else:
            self.game.set_window_visible(False)

        # Start the game
        self.game.init()

        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(3)

    # Function that is called on every Ai action (or step)
    def step(self, action):
        # Specify actions and take step
        actions = np.identity(3, np.uint8)
        reward = self.game.make_action(actions[action], 4)

        # Get necessary returns from client
        if self.game.get_state():
            image = self.game.get_state().screen_buffer
            image = self.grayscale(image)
            ammo = self.game.get_state().game_variables[0]
            info = {"ammo": ammo}
        else:
            image = np.zeros(self.observation_space.shape)
            info = 0

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
        return self.grayscale(state)
