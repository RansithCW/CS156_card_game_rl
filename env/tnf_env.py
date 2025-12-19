#TODO: Gym wrapper for TNF 
from pickle import NONE
import random
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
        super().__init__()
        
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
        self._state_buffer = np.zeros(214, dtype=np.float32)
        
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
        while self.game.current_player != self.agent_id and len(self.game.current_trick) < 8: # just for safe
            pid = self.game.current_player
            card = self._get_opponent_action(pid, self.game)
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
                reward += points*0.01
            else:
                reward -= points*0.01 # slightly higher penalty for losing pts
                
            if self.verbose and cards is not None:
                resolved_tricks.append({
                    "winnder": winner,
                    "cards": [str(index_to_card(c_id)) for c_id, _ in cards],
                    "points": points
                })
            
            if self.game.is_terminal():
                self.done = True
                # Final reward adjustment based on total points
                reward += 10.0 if self.game.team_points[self.agent_id % 2] > self.game.team_points[1 - (self.agent_id % 2)] else -15.0
                # penalty for losing, vs. small bonus for winning
                
        return reward, resolved_tricks

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")
        
        # if action not in self.game.legal_actions(self.agent_id):
        #     # Illegal action → strong negative reward + terminate
        #     return (
        #         self._get_obs(self.agent_id),
        #         -50.0, # negative reward
        #         True, # done
        #         False,
        #         {"action_mask": self.action_masks(),
        #          "resolved_tricks": []},
        #     )
            
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
            card = self._get_opponent_action(pid, self.game)
            self.game.play_card(pid, card)
            
            r, tricks = self._process_trick_resolution()
            total_reward += r
            resolved_tricks.extend(tricks)

        # 4. Final state check
        obs = self._get_obs(self.agent_id)
        info = {"action_mask": self.action_masks(self.agent_id),
                "resolved_tricks": resolved_tricks}
        
        return obs, total_reward, self.done, False, info
    
    def action_masks(self, agent_idx=None):
        """
        Returns action mask for current legal actions 
        as a numpy array of shape (32,), dtype int8
        """
        if agent_idx is None:
            agent_idx = self.agent_id
        mask = np.zeros(32, dtype=bool)
        for action_idx in self.game.legal_actions(agent_idx):
            mask[action_idx] = True
        return mask
    
    def _get_obs(self, player_id):
        # Use 'slices' to update the buffer instead of concatenating
        # This writes directly to the existing memory        
        game = self.game
        partner_id = (player_id + 2) % 4
        
        # 0. Reset buffer (Ensure size is 214 in __init__)
        # New size logic: 150 + 64 (for split trick encoding) = 214
        self._state_buffer.fill(0.0)
        
        # 1. Hand encoding [0:31]
        self._state_buffer[0:32] = game.current_hand_masks[player_id]
        
        # 2. Split Trick Encoding [32:95]
        # [32:63] Cards played by Partner
        # [64:95] Cards played by Opponents (either of the two)
        for c_idx, p_idx in game.current_trick:
            if p_idx == partner_id:
                self._state_buffer[32 + c_idx] = 1.0
            elif p_idx != player_id: # It's an opponent
                self._state_buffer[64 + c_idx] = 1.0
            
        # 3. Trick Metadata [96:159]
        if game.current_trick:
            winner, high_card_idx = game.current_trick_winner()
            
            # [96:127] Highest Card on table bitmask (Critical for 'beating' current high)
            self._state_buffer[96 + high_card_idx] = 1.0
            
            # [128:131] Lead Suit one-hot
            lead_suit_idx = game.current_trick[0][0] // 8
            self._state_buffer[128 + lead_suit_idx] = 1.0
            
            # [140] Partner winning flag (Explicit "don't trump your partner" signal)
            if winner == partner_id:
                self._state_buffer[140] = 1.0
                
            # [209] Trick winner ID (normalized)
            self._state_buffer[209] = winner / 3.0

        # 4. Trump [132:135] and Position [136:139]
        self._state_buffer[132 + game.trump_suit_idx] = 1.0
        self._state_buffer[136 + len(game.current_trick)] = 1.0

        # 5. Trick Number [141]
        self._state_buffer[141] = game.trick_number / 8.0

        # 6. Played Cards History (Memory) [142:173]
        self._state_buffer[142:174] = game.played_cards_mask

        # 7. Points [174:176] - Rescaled to 152 for stronger signal
        # 152 is half of 304; usually team pts don't cross this unless crushing
        our_pts = game.team_points[player_id % 2]
        opp_pts = game.team_points[1 - (player_id % 2)]
        self._state_buffer[174] = our_pts / 152.0
        self._state_buffer[175] = opp_pts / 152.0
        self._state_buffer[176] = (our_pts - opp_pts) / 100.0 # Raw diff signal
        
        # 8. Cards Remaining Per Suit [177:180]
        played_per_suit = game.played_cards_mask.reshape(4, 8).sum(axis=1)
        self._state_buffer[177:181] = (8.0 - played_per_suit) / 8.0        

        return self._state_buffer.copy()    
    def update_opponents(self, model_paths):
        """Called by the training callback to swap opponent logic to a frozen model."""
        from sb3_contrib import MaskablePPO
        from agents.rl_agent import RLAgent
        
        # randomly choose one of provided models
        opponents = random.choices(model_paths, k=3)
            
        for i in range(1, 4):
            model_path = opponents[i - 1]
            # Load the latest snapshot
            frozen_model = MaskablePPO.load(model_path)
            
            # Wrap it in our RLAgent helper (using deterministic=False for variety)
            new_rl_opponent = RLAgent(frozen_model, deterministic=False)
            
            # Update your internal opponent list (Players 1, 2, 3)
            # Note: Player 0 is the one being trained
            self.opponents[i] = new_rl_opponent
        
        if self.verbose:
            print(f"--- Opponents updated to snapshot: {opponents} ---")

    def _get_opponent_action(self, player_id, obs):
        """Helper to get action from whichever opponent type is currently set."""
        
        opponent = self.opponents[player_id]
        # Fallback to existing Random/Greedy logic
        if hasattr(opponent, 'model'): 
                # 1. RL Agent needs the numeric observation vector
                obs = self._get_obs(player_id) 
                # 2. It also needs the legal moves mask
                mask = self.action_masks(player_id)
                return opponent.select_action(obs, action_mask=mask)
            
        else:
            # 3. Greedy/Random Agents need the raw Game Engine
            # (Using your original GreedyAgent signature: select_action(game, id))
            return opponent.select_action(self.game, player_id)