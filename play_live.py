import sys
import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import DQN

ENV_ID = "FlappyBird-v0"
model_path = sys.argv[1] if len(sys.argv) > 1 else "logs/dqn_flappy/dqn_flappy_model.zip"

# create a human-rendering env (opens a window)
env = gym.make(ENV_ID, render_mode="human", use_lidar=True)
model = DQN.load(model_path)

obs, _ = env.reset()
try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
finally:
    env.close()