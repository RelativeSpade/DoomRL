from vizdoom import *
import random
import time
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import cv2
import os
from stable_baselines3.common.callbacks import BaseCallback

from Script.TrainAndLoggingCallback import TrainAndLoggingCallback
from stable_baselines3.common import env_checker

from Script.VizDoomGym import VizDoomGym

CHECKPOINT_DIR = '../train/'
LOG_DIR = '../logs/'

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

env = VizDoomGym(render=False)

env_checker.check_env(env)