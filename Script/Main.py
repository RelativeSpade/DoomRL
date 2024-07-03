from stable_baselines3 import PPO
from Script.GetLatestModel import get_latest_model
from Script.TrainAndLoggingCallback import TrainAndLoggingCallback
from Script.VizDoomGym import VizDoomGym

CHECKPOINT_DIR = '../train/'
LOG_DIR = '../logs/'

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

env = VizDoomGym(False, '../Vizdoom/scenarios/deadly_corridor (1).cfg')

latest_model_path, latest_n_calls = get_latest_model(CHECKPOINT_DIR)
if latest_model_path is None:
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=4096)
else:
    model = PPO.load(latest_model_path, env, tensorboard_log=LOG_DIR)
    callback.n_calls = latest_n_calls

model.learn(total_timesteps=60000, callback=callback)
