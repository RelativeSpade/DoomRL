import os
import re
from stable_baselines3 import PPO
from Script.TrainAndLoggingCallback import TrainAndLoggingCallback
from Script.VizDoomGym import VizDoomGym

CHECKPOINT_DIR = '../train/'
LOG_DIR = '../logs/'

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

env = VizDoomGym(False, '../Vizdoom/scenarios/defend_the_center.cfg')


def get_latest_model(checkpoint_dir):
    pattern = re.compile(r'Doom(\d+)')
    checkpoint_files = [
        (file, int(pattern.search(file).group(1)))
        for file in os.listdir(checkpoint_dir)
        if pattern.search(file)
    ]
    if not checkpoint_files:
        return None, 0

    latest_checkpoint = max(checkpoint_files, key=lambda x: x[1])
    print('Latest checkpoint:', latest_checkpoint[0])
    return os.path.join(checkpoint_dir, latest_checkpoint[0]), latest_checkpoint[1]


latest_model_path, latest_n_calls = get_latest_model(CHECKPOINT_DIR)
if latest_model_path is None:
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=4096)
else:
    model = PPO.load(latest_model_path, env, tensorboard_log=LOG_DIR)
    callback.n_calls = latest_n_calls

model.learn(total_timesteps=100000, callback=callback)
