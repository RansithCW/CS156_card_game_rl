# seed 43 for reproducibility

import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)


import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.tnf_env import TNFEnv


def make_env():
    def _init():
        return TNFEnv(game_type='random', seed=SEED)
    return _init


def train():
    # Vectorized env (even 1 env is required by SB3)
    env = DummyVecEnv([make_env()])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cpu" # CPU Faster for PPO than GPU
    )

    model.learn(total_timesteps=200_000)

    os.makedirs("models", exist_ok=True)
    model.save("models/tnf_ppo")

    env.close()


if __name__ == "__main__":
    train()
