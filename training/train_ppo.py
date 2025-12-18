# seed 42 for reproducibility
import numpy as np
import os
import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from env.tnf_env import TNFEnv

SEED = 42
np.random.seed(SEED)

# Callback to handle self-play opponent updates
class SelfPlayCallback(BaseCallback):
    """Saves a model snapshot and updates env opponents every N steps."""
    def __init__(self, check_freq, save_path):
        super().__init__()
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            path = os.path.join(self.save_path, f"snapshot_{self.n_calls}.zip")
            self.model.save(path)
            # Update all vectorized environments
            self.training_env.env_method("update_opponents", path)
        return True

def train(game_type='greedy', total_timesteps=200_000, load_path=None, device='cuda'):
    # 1. Create Env with the specific opponent type
    def make_env():
        return TNFEnv(game_type=game_type, seed=42)
    
    venv = DummyVecEnv([make_env])
        
    # Add Normalization
    # norm_obs: scales the 150-dim observation vector
    # norm_reward: scales the 0-304 rewards
    # Needed as tricks can vary widely from 0 to 71 pts
    env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Log directory for TensorBoard plots
    log_dir = f"./logs/tnf_{game_type}/"
    snapshot_dir = f"./models/snapshots_{game_type}/" # Folder for self-play history    
    
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

    # 3. Setup Self-Play Callback
    # check_freq: how often to update opponents (e.g., every 50,000 steps)
    self_play_callback = SelfPlayCallback(
        check_freq=50000, 
        save_path=snapshot_dir
    )

    # 4. Train with Callback
    model.learn(
        total_timesteps=total_timesteps, 
        reset_num_timesteps=False,
        tb_log_name=f"run_{total_timesteps}_steps",
        callback=self_play_callback  # <--- IMPORTANT
    )

    # 5. SAVE final model and normalization stats
    save_dir = f"models/stage_{game_type}"
    os.makedirs(save_dir, exist_ok=True)
    
    model.save(os.path.join(save_dir, f"tnf_{game_type}_final"))
    env.save(os.path.join(save_dir, "vec_normalize.pkl")) # Save stats for evaluation
    
    print(f"Training Complete.")
    env.close()

if __name__ == "__main__":
    # Example: Start with random
    train(game_type='greedy', total_timesteps=500_000)