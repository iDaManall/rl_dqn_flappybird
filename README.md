# RL_DQN_flappybird

This project trains an autonomous agent to play Flappy Bird using a Gymnasium-compatible (`flappy-bird-gymnasium`) environment and a Deep Q‑Learning (DQN) algorithm (with Stable‑Baselines3). The goal is simple: teach an agent to keep the bird alive and pass pipes by maximizing a reward signal provided by the environment (small reward per frame alive, bonus for passing pipes, penalty for collisions). The project emphasizes reproducible training, clear logging of progress, and simple evaluation so that learning behavior can be demonstrated with charts and short gameplay. The end result is a saved DQN policy that can be evaluated deterministically and compared across training checkpoints to show improvement.

## Quick overview
- Environment: `FlappyBird-v0` (numeric observations; LIDAR or pipe/player features)
- Algorithm: DQN (Stable‑Baselines3, `MlpPolicy`)
- Logging: TensorBoard + `logs/monitor.csv`
- Playback: `play_live.py`

## Setup
1. Create and activate conda env (Miniforge/conda):
```bash
conda create -n rlflappy python=3.10 -y
conda activate rlflappy
conda install -c conda-forge pygame opencv pkg-config -y
python3 -m pip install --upgrade pip
python3 -m pip install git+https://github.com/markub3327/flappy-bird-gymnasium.git stable-baselines3 gymnasium tensorboard
```

## Run training
```bash
conda activate rlflappy
python3 rl_flappybird.py
```
- Edit `TOTAL_TIMESTEPS` in `rl_flappybird.py` to change run length.
- Checkpoints and evals are saved to `logs/dqn_flappy/`.

## View TensorBoard graphs:
```bash
# start tensorboard to see the training charts from logs/dqn_flappy:
python3 -m tensorboard.main --logdir logs/dqn_flappy
```

## Play live
```bash
python3 play_live.py logs/dqn_flappy/dqn_flappy_model.zip
```

## Resume training from checkpoint (example)
```python
from stable_baselines3 import DQN
model = DQN.load("logs/dqn_flappy/dqn_ckpt_50000_steps.zip", env=vec_env)
model.learn(total_timesteps=200000)
```

## Notes / Reproducibility
- On macOS M1, an incompatible TensorFlow wheel caused segfaults during development; remove `tensorflow` from the env if you encounter crashes (assuming you have it in the env).


## License & references
- flappy-bird-gymnasium: https://github.com/markub3327/flappy-bird-gymnasium
- Stable‑Baselines3: https://github.com/DLR-RM/stable-baselines3
