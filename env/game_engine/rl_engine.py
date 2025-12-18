from pyparsing import nums
from env.game_engine.constants import (
    SUITS,
    RANKS,
    POINT_LOOKUP,
)
from env.game_engine.game_engine import Card
import random
import numpy as np

NUM_RANKS = len(RANKS)
NUM_SUITS = len(SUITS)
NUM_CARDS = NUM_SUITS * NUM_RANKS  # should be 32


def card_to_index(card: Card) -> int:
    """
    Maps a Card to an integer in [0, 31]
    """
    return SUITS.index(card.suit) * NUM_RANKS + RANKS.index(card.rank)


def index_to_card(idx: int) -> Card:
    """
    Maps an integer in [0, 31] back to a Card
    """
    suit = SUITS[idx // NUM_RANKS]
    rank = RANKS[idx % NUM_RANKS]
    return Card(suit, rank)


class ThreeNoughtFourGame: # have to define game without async for training
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.trump_suit_idx = self.rng.randint(0, NUM_SUITS - 1)
        self.trump_suit = SUITS[self.trump_suit_idx]

        self.hands = [list() for _ in range(4)]
        self.lead_player = 0
        self.current_trick = []  # [(card_idx, player_id)]
        
        self.trick_number = 0
        
        self.team_points = [0, 0] # Team 0 (P0, P2) and Team 1 (P1, P3)
        self.played_cards_mask = np.zeros(32, dtype=np.float32) # Pre-allocated
        self.current_hand_masks = [np.zeros(32, dtype=np.float32) for _ in range(4)]
        
        self._deal()

        
    def _deal(self):
        deck = list(range(NUM_CARDS))
        self.rng.shuffle(deck)

        for i in range(4):
            self.hands[i] = deck[i * 8:(i + 1) * 8]
            # update masks using indices
            for card_idx in self.hands[i]:
                self.current_hand_masks[i][card_idx] = 1.0


    def legal_actions(self, player_id: int) -> list[int]:
        hand = self.hands[player_id]

        if not self.current_trick:
            return hand  # any card at start of trick

# Fast Suit Check: index // 8 gives the suit_idx
        lead_card_idx = self.current_trick[0][0]
        lead_suit_idx = lead_card_idx // 8
        
        follow = [c_idx for c_idx in hand if (c_idx // 8) == lead_suit_idx]
        if follow:
            return follow

        # TODO: Double check if we need to force trumping if can't answer to lead suit
        
        # trump_cards = [c for c in hand if c.suit == self.trump_suit]
        # if trump_cards:
        #     return [card_to_index(c) for c in trump_cards]

        return hand


    def play_card(self, player_id: int, card_idx: int):

        assert card_idx in self.legal_actions(player_id), "Illegal Move"
        
        # 1. Update hand mask (Remove card)
        self.current_hand_masks[player_id][card_idx] = 0.0
        
        # 2. Update played cards mask (Add card)
        self.played_cards_mask[card_idx] = 1.0
        
        self.hands[player_id].remove(card_idx)
        
        self.current_trick.append((card_idx, player_id))

    def _get_card_strength(self, card_idx):
        """Helper to determine trick winner using integer math"""
        suit_idx = card_idx // 8
        rank_idx = card_idx % 8
        
        lead_suit_idx = self.current_trick[0][0] // 8
        
        # Trump gets highest base weight
        if suit_idx == self.trump_suit_idx:
            return 50 + POINT_LOOKUP[rank_idx]
        # Lead suit gets medium base weight
        elif suit_idx == lead_suit_idx:
            return POINT_LOOKUP[rank_idx]
        # Others have no winning power
        return 0

    def resolve_trick(self, verbose=False):

        winner_card, winner = max(
            self.current_trick,
            key=lambda x: self._get_card_strength(x[0])
        )

        points = sum(POINT_LOOKUP[c_idx % 8] for c_idx, _ in self.current_trick)
        
        # Update team points incrementally
        self.team_points[winner % 2] += points
        
        # State cleanup
        cards_in_trick = list(self.current_trick) if verbose else None
        self.current_trick = []
        self.lead_player = winner
        self.trick_number += 1

        return winner, points, cards_in_trick
    
    def current_trick_winner(self):
        """
        Returns player_id and highest card of current trick without resolving
        Assumes current_trick is non-empty
        """
        winnder_card_idx, winner = max(
            self.current_trick,
            key=lambda x: self._get_card_strength(x[0])
        )
        return winner, winnder_card_idx

    
    def is_terminal(self):
        return all(len(hand) == 0 for hand in self.hands)
    
    @property
    def current_player(self) -> int:
        if not self.current_trick:
            return self.lead_player
        return (self.lead_player + len(self.current_trick)) % 4


