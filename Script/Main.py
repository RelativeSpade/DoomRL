from stable_baselines3 import PPO

from Script.TrainAndLoggingCallback import TrainAndLoggingCallback
from Script.VizDoomGym import VizDoomGym

CHECKPOINT_DIR = '../train/'
LOG_DIR = '../logs/'

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

env = VizDoomGym(False, '../Vizdoom/scenarios/defend_the_center.cfg')

model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=256)

model.learn(total_timesteps=100000, callback=callback)