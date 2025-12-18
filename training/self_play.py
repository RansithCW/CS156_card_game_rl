import os
from stable_baselines3.common.callbacks import BaseCallback

class SelfPlayCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(SelfPlayCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Every 'check_freq' steps, update the competition
        if self.n_calls % self.check_freq == 0:
            snapshot_name = f"self_play_v{self.n_calls}.zip"
            model_path = os.path.join(self.save_path, snapshot_name)
            
            # 1. Save the current state of the "main" agent
            self.model.save(model_path)
            
            # 2. Tell the environments to load this model for the opponents
            # We use env_method to reach into the vectorized environments
            self.training_env.env_method("update_opponents", model_path)
            
            if self.verbose > 0:
                print(f"Step {self.n_calls}: Snapshot saved and opponents updated!")
                
        return True
    

def train_self_play(total_timesteps=1_000_000):
    # Create the environment
    env = DummyVecEnv([lambda: TNFEnv(game_type='mixed')])

    # Initialize MaskablePPO
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        ent_coef=0.01,
        tensorboard_log="./logs/self_play/"
    )

    # Setup the callback
    # This will update the opponents every 50k steps
    self_play_cb = SelfPlayCallback(
        check_freq=50000, 
        save_path="./models/snapshots/"
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps, 
        callback=self_play_cb
    )
    
    model.save("final_self_play_agent")