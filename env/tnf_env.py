#TODO: Gym wrapper for TNF 
import re
from tabnanny import verbose
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.game_engine.rl_engine import ThreeNoughtFourGame, card_to_index
from env.game_engine.state_encoder import encode_state
from env.game_engine.constants import SUITS

class TNFEnv(gym.Env):
    def __init__(self, seed=None, game_type='mixed', verbose=False):
        super(TNFEnv, self).__init__()
        
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
            
        # Create a dummy game to calculate the actual shape once
        example_obs = self._get_obs(0)
        
        self.observation_space = spaces.Box(
            low = 0.0,
            high = 1.0,
            shape=example_obs.shape, # changes if state encoding changes
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(32)  # 32 possible cards to play
        
        self.game = ThreeNoughtFourGame(seed=seed) # None ?
        self.done = False
        
        # Pre-allocate the entire state vector once
        self._state_buffer = np.zeros(self.observation_space.shape, dtype=np.float32)


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
        
        info = {"action_mask": self._action_mask(),
                "resolved_tricks": tricks,}
        
        return obs, info
    
    
    def _process_trick_resolution(self):
        """Helper to resolve tricks and assign rewards."""
        reward = 0.0
        resolved_tricks = []
        
        while len(self.game.current_trick) >= 4:
            # record trick before resolving
            trick_cards = [str(c) for c, _ in self.game.current_trick]
            
            winner, points, cards = self.game.resolve_trick(verbose=self.verbose)
            
            # Team Scoring
            # If agent or partner (team 0 & 2) wins, add pts
            if winner % 2 == self.agent_id % 2:
                reward += points
            else:
                reward -= points
                
            if cards is not None:
                resolved_tricks.append({
                    "winnder": winner,
                    "cards": [str(c) for c, _ in cards],
                    "points": points
                })
            
            if self.game.is_terminal():
                self.done = True
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
                {"action_mask": self._action_mask(),
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
        info = {"action_mask": self._action_mask(),
                "resolved_tricks": resolved_tricks}
        

        return obs, total_reward, self.done, False, info
    
    def _action_mask(self):
        """
        Returns action mask for current legal actions 
        as a numpy array of shape (32,), dtype int8
        """
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for a in self.game.legal_actions(self.agent_id):
            mask[a] = 1
        return mask
    
    def _get_obs(self, player_id):
        # Use 'slices' to update the buffer instead of concatenating
        # This writes directly to the existing memory
        game = self.game
        
        # Reset buffer at start of each observation
        self._state_buffer.fill(0.0)
        
        # Hand encoding [32]
        self._state_buffer[0:32] = game.current_hand_masks[player_id]
        
        # Current Trick [32]
        for card, _ in game.current_trick:
            self._state_buffer[32 + card_to_index(card)] = 1.0
            
        # Highest Card in Trick (32)
        if game.current_trick:
            winner, high_card = game.current_trick_winner()
            self._state_buffer[64 + card_to_index(high_card)] = 1.0
            
            # 108: Partner winning flag (1) - Updating this slice here saves a second winner check
            partner_id = (player_id + 2) % 4
            if winner == partner_id:
                self._state_buffer[108] = 1.0

        # Lead Suit one-hot (4)
        if game.current_trick:
            lead_suit = game.current_trick[0][0].suit
            self._state_buffer[96 + SUITS.index(lead_suit)] = 1.0

        # Trump Suit one-hot (4)
        self._state_buffer[100 + SUITS.index(game.trump_suit)] = 1.0

        # Position in trick one-hot (4)
        self._state_buffer[104 + len(game.current_trick)] = 1.0

        # Trick number (1)
        self._state_buffer[109] = game.trick_number / 8.0

        # Played Cards (32)
        self._state_buffer[110:142] = game.played_cards_mask

        # Points (3)
        our_pts = game.team_points[player_id % 2]
        opp_pts = game.team_points[1 - (player_id % 2)]
        self._state_buffer[142] = our_pts / 304.0
        self._state_buffer[143] = opp_pts / 304.0
        self._state_buffer[144] = (our_pts - opp_pts) / 304.0

        return self._state_buffer.copy()
