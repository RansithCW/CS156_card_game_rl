# seed 42 for reproducibility
import numpy as np
import os
import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.tnf_env import TNFEnv

SEED = 42
np.random.seed(SEED)



def train(game_type='random', total_timesteps=200_000, load_path=None, device='cpu'):
    # 1. Create Env with the specific opponent type
    def make_env():
        return TNFEnv(game_type=game_type, seed=42)
    
    env = DummyVecEnv([make_env])

    # Log directory for TensorBoard plots
    log_dir = f"./logs/tnf_{game_type}/"
        
    # 2. LOAD or CREATE model
    if load_path and os.path.exists(load_path):
        print(f"--- Loading existing model from {load_path} ---")
        model = MaskablePPO.load(load_path, env=env, device=device)
    else:
        print(f"--- Creating NEW MaskablePPO model for {game_type} training ---")
        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            ent_coef=0.01, # Keep exploration high during early stages
            learning_rate=3e-4,
            tensorboard_log=log_dir,
            device=device
        )

    # 3. Train
    model.learn(
        total_timesteps=total_timesteps, reset_num_timesteps=False,
        tb_log_name=f"run_{total_timesteps}_steps"
        )

    # 4. SAVE
    save_dir = f"models/stage_{game_type}"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"tnf_{game_type}_{total_timesteps}_steps")
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    env.close()

if __name__ == "__main__":
    # Example: Start with random
    # train(game_type='random', total_timesteps=100_000) # loss started inc after 75k timesteps
    
    # Example: Continue same model against mixed
    train(game_type='mixed', total_timesteps=150_000, load_path="models/stage_random/tnf_random_100000_steps.zip")
    
    # Example: Continue same model against greedy
    train(game_type='greedy', total_timesteps=300_000, load_path="models/stage_mixed/tnf_mixed_150000_steps.zip")