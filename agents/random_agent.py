import numpy as np

class RandomAgent:
    def select_action(self, game, player_id: int) -> int:
        legal_actions = game.legal_actions(player_id)
        return np.random.choice(legal_actions)