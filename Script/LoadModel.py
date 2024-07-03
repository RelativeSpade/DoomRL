import time
from stable_baselines3 import PPO
from Script.GetLatestModel import get_latest_model
from Script.VizDoomGym import VizDoomGym


# directory of model
CHECKPOINT_DIR = '../train/'
load, _ = get_latest_model(CHECKPOINT_DIR)
# load model from disk
model = PPO.load(load)
# create rendered game
env = VizDoomGym(True, '../Vizdoom/scenarios/deadly_corridor (1).cfg')

# evaluate mean score for 10 games
total_reward = 0
for episode in range(1):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        time.sleep(0.05)
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(episode, total_reward))
    print('Mean: {}'.format(total_reward / (episode+1)))
    time.sleep(2)
