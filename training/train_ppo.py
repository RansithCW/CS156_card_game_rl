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


def train(game_type='random', total_timesteps=200_000, load_path=None, device='cpu'):
    # 1. Create Env with the specific opponent type
    def make_env():
        return TNFEnv(game_type=game_type, seed=42)
    
    env = DummyVecEnv([make_env])

    # 2. LOAD or CREATE model
    if load_path and os.path.exists(load_path):
        print(f"--- Loading existing model from {load_path} ---")
        model = PPO.load(load_path, env=env, device=device)
    else:
        print(f"--- Creating NEW model for {game_type} training ---")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            ent_coef=0.01, # Keep exploration high during early stages
            device=device
        )

    # 3. Train
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

    # 4. SAVE with a clear name
    save_dir = f"models/stage_{game_type}"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"tnf_{game_type}_{total_timesteps}steps")
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    env.close()

if __name__ == "__main__":
    # Example: Start with random
    train(game_type='random', total_timesteps=75_000) # loss started inc after 75k timesteps
    
    # Example: Continue same model against mixed
    # train(game_type='mixed', total_timesteps=300_000, load_path="models/stage_random/tnf_random_300000steps.zip")
    
    # Example: Continue same model against greedy
    # train(game_type='greedy', total_timesteps=300_000, load_path="models/stage_random/tnf_mixed_300000steps.zip")