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
        self.device = self.model.device

    def select_action(self, obs, action_mask=None) -> int:
        """
        obs: np.ndarray, shape (obs_dim,)
        action_mask: np.ndarray of shape (32,), 1 = legal, 0 = illegal
        """
        obs_tensor = torch.as_tensor(obs[None], dtype=torch.float32, device=self.device)

        if action_mask is None:
            action, _ = self.model.predict(
                obs_tensor, deterministic=self.deterministic
            )
            return int(action[0])

        # --- Masked action selection ---
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)

            logits = dist.distribution.logits # shape [1, 32]

            # Apply mask
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device) #[32]
            masked_logits = logits.masked_fill(~mask, -1e8)

            if self.deterministic:
                # Greedy selection
                action = torch.argmax(masked_logits, dim=-1)
            else:
                # Stochastic selection: Re-create distribution with masked logits
                # This ensures the probabilities sum to 1 among legal moves
                new_dist = torch.distributions.Categorical(logits=masked_logits)
                action = new_dist.sample()
            
            return int(action.item())
