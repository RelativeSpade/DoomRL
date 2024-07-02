from vizdoom import *
import random
import time
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import cv2

actions = np.identity(3, dtype=int)

