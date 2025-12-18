#TODO: Gym wrapper for TNF 
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.game_engine.rl_engine import ThreeNoughtFourGame
from env.game_engine.state_encoder import encode_state

class TNFEnv(gym.Env):
    def __init__(self, seed=None, game_type='mixed'):
        super(TNFEnv, self).__init__()
        
        self.agent_id = 0 # single agent env
        
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
        
        self.observation_space = spaces.Box(
            low = 0.0,
            high = 1.0,
            shape=(110,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(32)  # 32 possible cards to play
        
        self.game = ThreeNoughtFourGame(seed=seed) # None ?
        self.done = False


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.game = ThreeNoughtFourGame(seed=seed)
        self.done = False
        
        # Autoplay until agent's turn
        while self.game.current_player != self.agent_id:
            pid = self.game.current_player
            card = self.opponents[pid].select_action(self.game, pid)
            self.game.play_card(pid, card)

            if len(self.game.current_trick) == 4:
                self.game.resolve_trick()
        
        obs = encode_state(self.game, self.agent_id)
        info = {"action_mask": self._action_mask()}
        
        return obs, info
    
    
    def _process_trick_resolution(self):
        """Helper to resolve tricks and assign rewards."""
        reward = 0.0
        while len(self.game.current_trick) >= 4:
            winner, points = self.game.resolve_trick()
            # If agent or agent's partner (team 0 & 2) wins
            if winner % 2 == self.agent_id % 2:
                reward += points
            else:
                reward -= points
            
            if self.game.is_terminal():
                self.done = True
        return reward

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")
        
        if action not in self.game.legal_actions(self.agent_id):
            # Illegal action → strong negative reward + terminate
            return (
                encode_state(self.game, self.agent_id),
                -50.0, # negative reward
                True, # done
                False,
                {"action_mask": self._action_mask()},
            )
            
        total_reward = 0.0

        # 1. Agent plays
        self.game.play_card(self.agent_id, action)
        
        # 2. Check if agent's card finished the trick
        total_reward += self._process_trick_resolution()

        # 3. Autoplay opponents until it's agent's turn or game ends
        while not self.done and self.game.current_player != self.agent_id:
            pid = self.game.current_player
            card = self.opponents[pid].select_action(self.game, pid)
            self.game.play_card(pid, card)
            
            total_reward += self._process_trick_resolution()

        # 4. Final state check
        obs = encode_state(self.game, self.agent_id)
        info = {"action_mask": self._action_mask()}
        

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
