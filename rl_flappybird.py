# Training a DQN agent on FlappyBird-v0 from flappy-bird-gymnasium.

import os
import gymnasium as gym
import flappy_bird_gymnasium  # ensures the FlappyBird-v0 env is registered
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# where logs, tensorboard files and checkpoints will be stored
LOGDIR = "logs/dqn_flappy"
os.makedirs(LOGDIR, exist_ok=True)

ENV_ID = "FlappyBird-v0" # from the flappy_bird_gymnasium README

# creates and wraps a single environment instance.
def make_env():
    # use_lidar=True for 180 numeric lidar readings (floats), or False for pipe/player features
    env = gym.make(ENV_ID, render_mode=None, use_lidar=True)
    # monitor logs per-episode rewards/lengths to monitor.csv.
    env = Monitor(env, filename=os.path.join(LOGDIR, "monitor.csv"))
    return env

# Stable-Baselines3 expects a vectorized env; DummyVecEnv is a single-process wrapper.
vec_env = DummyVecEnv([make_env])

checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=LOGDIR, name_prefix="dqn_ckpt")
eval_env_cb = DummyVecEnv([make_env])
eval_callback = EvalCallback(eval_env_cb, best_model_save_path=LOGDIR,
                            log_path=LOGDIR, eval_freq=25_000,
                            deterministic=True, render=False)

model = DQN(
    "MlpPolicy",       # MlpPolicy = simple feed-forward network for numeric inputs.
    vec_env,
    verbose=1,
    buffer_size=50_000,  # replay buffer capacity
    learning_rate=1e-4,  
    batch_size=64,
    learning_starts=1_000,  # number of steps before learning begins
    train_freq=4,   # how often (in env steps) to perform a gradient update
    target_update_interval=1_000,  # how often to sync target network
    gamma=0.99,
    tensorboard_log=LOGDIR, # enables tensorboard logging
)

TOTAL_TIMESTEPS = 3_000_000  # number of environment steps (frames) to train for.

# Run training. This runs the full training loop and logs to tensorboard/monitor.csv.
# model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="dqn_flappy")
model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="dqn_flappy",
            callback=[checkpoint_callback, eval_callback])

# Save final model
model_path = os.path.join(LOGDIR, "dqn_flappy_model")
model.save(model_path)
print("Saved model to", model_path)

# Quick evaluation: run a few deterministic episodes and report mean + std of returns
eval_env = make_env()
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"Evaluation mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
eval_env.close()