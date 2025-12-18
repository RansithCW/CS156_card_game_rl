#TODO: implement state_encoder.py for env package
from matplotlib import tri
import numpy as np

from env.game_engine.constants import RANKS, SUITS, POINTS
from env.game_engine.rl_engine import NUM_SUITS, card_to_index, index_to_card, NUM_CARDS

NUM_SUITS = len(SUITS)


def encode_state(game, player_id: int) -> np.ndarray:
    """
    Encodes the game state from the perspective of player_id.

    Returns:
        np.ndarray of shape (114,), dtype float32
    """

    # --- 1. Hand encoding (32) ---
    hand_vec = np.zeros(NUM_CARDS, dtype=np.float32)
    for card in game.hands[player_id]:
        hand_vec[card_to_index(card)] = 1.0
    
    # --- 2. Current trick cards (32) ---
    trick_vec = np.zeros(NUM_CARDS, dtype=np.float32)
    for card, _ in game.current_trick:
        trick_vec[card_to_index(card)] = 1.0
    
    # --- 3. Highest Card in trcik (32) ---
    highest_vec = np.zeros(NUM_CARDS, dtype=np.float32)
    if game.current_trick:
        high_card = game.current_trick_winner()[1]
        highest_vec[card_to_index(high_card)] = 1.0
    
    # --- 4. Lead Suit one-hot (4) ---
    lead_suit_vec = np.zeros(NUM_SUITS, dtype=np.float32)
    if game.current_trick:
        lead_suit = game.current_trick[0][0].suit
        lead_suit_vec[SUITS.index(lead_suit)] = 1.0
    
    # --- 5. Trump Suit one-hot (4) ---
    trump_vec = np.zeros(NUM_SUITS, dtype=np.float32)
    trump_vec[SUITS.index(game.trump_suit)] = 1.0
    
    # --- 6. Position in trick (4) ---
    position_vec = np.zeros(4, dtype=np.float32)
    position_vec[len(game.current_trick)] = 1.0
    
    # --- 7. Partner winning flag (1) ---
    partner_id = (player_id + 2) % 4
    partner_winning = 0.0
    if game.current_trick:
        winner = game.current_trick_winner()[0]
        if winner == partner_id:
            partner_winning = 1.0
    partner_vec = np.array([partner_winning], dtype=np.float32)
    
    # --- 8. Trick number (1) ---
    trick_number_vec = np.array([game.trick_number / 8.0], dtype=np.float32)
    
    # --- Concatenate all parts to one state vector ---
    state = np.concatenate([
        hand_vec,           # 32
        trick_vec,          # 32
        highest_vec,        # 32
        lead_suit_vec,      # 4
        trump_vec,          # 4
        position_vec,       # 4
        partner_vec,        # 1
        trick_number_vec,    # 1
    ])
    
    return state # size (110,)