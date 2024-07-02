from vizdoom import *
import random
import time
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import cv2
import os
from stable_baselines3.common.callbacks import BaseCallback

CHECKPOINT_DIR = '../train/'
LOG_DIR = '../logs/'

