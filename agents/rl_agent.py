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
        obs: Can be a numpy array or the Game object (we handle both)
        action_mask: np.ndarray of shape (32,), 1 = legal, 0 = illegal
        """
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)

        obs_tensor = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _ = self.model.predict(
                obs_tensor.cpu().numpy(),
                action_masks=action_mask,
                deterministic=self.deterministic
                )

        return int(action[0] if isinstance(action, np.ndarray) else action)
        