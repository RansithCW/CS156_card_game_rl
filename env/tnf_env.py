#TODO: Gym wrapper for TNF 
import re
from tabnanny import verbose
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.game_engine.rl_engine import ThreeNoughtFourGame, card_to_index, index_to_card
from env.game_engine.state_encoder import encode_state
from env.game_engine.constants import SUITS

class TNFEnv(gym.Env):
    def __init__(self, seed=None, game_type='mixed', verbose=False):
        super(TNFEnv, self).__init__()
        
        self.game = ThreeNoughtFourGame(seed=seed) # None for random seed
        
        self.agent_id = 0 # single agent env
        
        self.verbose = verbose
        
        self.opponents = {
            1: GreedyAgent(),
            2: RandomAgent(),
            3: GreedyAgent(),
        }
        if game_type == 'random':
            self.opponents[1] = RandomAgent()
            self.opponents[3] = RandomAgent()
        elif game_type == 'greedy':
            self.opponents[2] = GreedyAgent()
            
        # Pre-allocate the entire state vector once
        self._state_buffer = np.zeros(150, dtype=np.float32)
        
        # Create a dummy game to calculate the actual shape once
        example_obs = self._get_obs(0)
        
        self.observation_space = spaces.Box(
            low = 0.0,
            high = 1.0,
            shape=example_obs.shape, # changes if state encoding changes
            dtype=np.float32
        )
                
        self.action_space = spaces.Discrete(32)  # 32 possible cards to play
        
        self.done = False
        


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game = ThreeNoughtFourGame(seed=seed)
        self.done = False
        tricks = []
        
        # Autoplay until agent's turn
        while self.game.current_player != self.agent_id:
            pid = self.game.current_player
            card = self.opponents[pid].select_action(self.game, pid)
            self.game.play_card(pid, card)

            if len(self.game.current_trick) == 4:
                reward, tricks = self._process_trick_resolution()
        
        obs = self._get_obs(self.agent_id)
        
        info = {"action_mask": self.action_masks(),
                "resolved_tricks": tricks,}
        
        return obs, info
    
    
    def _process_trick_resolution(self):
        """Helper to resolve tricks and assign rewards."""
        reward = 0.0
        resolved_tricks = []
        
        while len(self.game.current_trick) >= 4:
            winner, points, cards = self.game.resolve_trick(verbose=self.verbose)
            
            # Team Scoring
            # If agent or partner (team 0 & 2) wins, add pts
            if winner % 2 == self.agent_id % 2:
                reward += points
            else:
                reward -= points
                
            if self.verbose and cards is not None:
                resolved_tricks.append({
                    "winnder": winner,
                    "cards": [str(index_to_card(c_id)) for c_id, _ in cards],
                    "points": points
                })
            
            if self.game.is_terminal():
                self.done = True
                # Final reward adjustment based on total points
                reward += 10.0 * (self.game.team_points[self.agent_id % 2] -154.0) / 304.0
                # give more reward for scoring more than half the points, increasin with higher pts
                
        return reward, resolved_tricks

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")
        
        if action not in self.game.legal_actions(self.agent_id):
            # Illegal action → strong negative reward + terminate
            return (
                self._get_obs(self.agent_id),
                -50.0, # negative reward
                True, # done
                False,
                {"action_mask": self.action_masks(),
                 "resolved_tricks": []},
            )
            
        total_reward = 0.0
        resolved_tricks = []

        # 1. Agent plays
        self.game.play_card(self.agent_id, action)
        
        # 2. Check if agent's card finished the trick
        r, tricks = self._process_trick_resolution()
        total_reward += r
        resolved_tricks.extend(tricks)

        # 3. Autoplay opponents until it's agent's turn or game ends
        while not self.done and self.game.current_player != self.agent_id:
            pid = self.game.current_player
            card = self.opponents[pid].select_action(self.game, pid)
            self.game.play_card(pid, card)
            
            r, tricks = self._process_trick_resolution()
            total_reward += r
            resolved_tricks.extend(tricks)

        # 4. Final state check
        obs = self._get_obs(self.agent_id)
        info = {"action_mask": self.action_masks(),
                "resolved_tricks": resolved_tricks}
        

        return obs, total_reward, self.done, False, info
    
    def action_masks(self):
        """
        Returns action mask for current legal actions 
        as a numpy array of shape (32,), dtype int8
        """
        mask = np.zeros(32, dtype=bool)
        for action_idx in self.game.legal_actions(self.agent_id):
            mask[action_idx] = True
        return mask
    
    def _get_obs(self, player_id):
        # Use 'slices' to update the buffer instead of concatenating
        # This writes directly to the existing memory        
        game = self.game
        
        # 0. Reset buffer (Ensure size is 150 in __init__)
        self._state_buffer.fill(0.0)
        
        # 1. Hand encoding [0:31]
        self._state_buffer[0:32] = game.current_hand_masks[player_id]
        
        # 2. Current Trick [32:63]
        for c_idx, _ in game.current_trick:
            self._state_buffer[32 + c_idx] = 1.0
            
        # 3. Merged Trick Logic: Winner, Highest Card, Partner Flag, Trick Winner ID
        if game.current_trick:
            winner, high_card_idx = game.current_trick_winner()
            
            # [64:95] Highest Card bitmask
            self._state_buffer[64 + high_card_idx] = 1.0
            
            # [108] Partner winning flag
            partner_id = (player_id + 2) % 4
            if winner == partner_id:
                self._state_buffer[108] = 1.0
                
            # [145] Trick winner ID (normalized)
            self._state_buffer[145] = winner / 3.0
            
            # [96:99] Lead Suit one-hot
            lead_suit_idx = game.current_trick[0][0] // 8
            self._state_buffer[96 + lead_suit_idx] = 1.0

        # 4. Trump [100:103] and Position [104:107]
        self._state_buffer[100 + game.trump_suit_idx] = 1.0
        self._state_buffer[104 + len(game.current_trick)] = 1.0

        # 5. Trick Number [109]
        self._state_buffer[109] = game.trick_number / 8.0

        # 6. Played Cards History [110:141]
        self._state_buffer[110:142] = game.played_cards_mask

        # 7. Points [142:144]
        our_pts = game.team_points[player_id % 2]
        opp_pts = game.team_points[1 - (player_id % 2)]
        self._state_buffer[142] = our_pts / 304.0
        self._state_buffer[143] = opp_pts / 304.0
        self._state_buffer[144] = (our_pts - opp_pts) / 304.0
        
        # 8. Cards Remaining Per Suit [146:149]
        # Counts cards not yet played in each suit
        # Reshape (32,) -> (4, 8), sum across the ranks, then normalize
        played_per_suit = game.played_cards_mask.reshape(4, 8).sum(axis=1)
        self._state_buffer[146:150] = (8.0 - played_per_suit) / 8.0        
        

        return self._state_buffer.copy()
