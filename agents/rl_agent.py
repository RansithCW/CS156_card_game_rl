import numpy as np
import torch

class RLAgent:
    def __init__(self, model, deterministic=True):
        """
        model: trained SB3 PPO model
        deterministic: whether to use deterministic actions (True for eval)
        """
        self.model = model
        self.deterministic = deterministic

    def select_action(self, obs, action_mask=None) -> int:
        """
        obs: np.ndarray, shape (obs_dim,)
        action_mask: np.ndarray of shape (32,), 1 = legal, 0 = illegal
        """
        obs = obs.reshape(1, -1)

        if action_mask is None:
            action, _ = self.model.predict(
                obs, deterministic=self.deterministic
            )
            return int(action)

        # --- Masked action selection ---
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).to(self.model.device)
            dist = self.model.policy.get_distribution(obs_tensor)

            logits = dist.distribution.logits.clone()

            mask = torch.as_tensor(action_mask).to(logits.device)
            logits[mask == 0] = -1e8

            action = torch.argmax(logits, dim=1)
            return int(action.item())
