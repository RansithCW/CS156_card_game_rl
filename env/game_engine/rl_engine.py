from env.game_engine.constants import (
    SUITS,
    RANKS,
    POINTS,
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
        self.trump_suit = self.rng.choice(SUITS)

        self.hands = [list() for _ in range(4)]

        self.lead_player = 0
        self.current_trick = []  # [(card, player_id)]
        
        self.trick_number = 0
        
        self.team_points = [0, 0] # Team 0 (P0, P2) and Team 1 (P1, P3)
        self.played_cards_mask = np.zeros(32, dtype=np.float32) # Pre-allocated
        self.current_hand_masks = [np.zeros(32, dtype=np.float32) for _ in range(4)]
        
        self._deal()
        # Initialize hand masks after dealing
        for pid in range(4):
            for card in self.hands[pid]:
                self.current_hand_masks[pid][card_to_index(card)] = 1.0
        
    def _deal(self):
        deck = [Card(s, r) for s in SUITS for r in RANKS]
        self.rng.shuffle(deck)

        for i in range(4):
            self.hands[i] = deck[i * 8:(i + 1) * 8]


    def legal_actions(self, player_id: int) -> list[int]:
        hand = self.hands[player_id]

        if not self.current_trick:
            return [card_to_index(c) for c in hand]  # any card at start of trick

        lead_suit = self.current_trick[0][0].suit

        follow = [c for c in hand if c.suit == lead_suit]
        if follow:
            return [card_to_index(c) for c in follow]

        # TODO: Double check if we need to force trumping if can't answer to lead suit
        
        # trump_cards = [c for c in hand if c.suit == self.trump_suit]
        # if trump_cards:
        #     return [card_to_index(c) for c in trump_cards]

        return [card_to_index(c) for c in hand]


    def play_card(self, player_id: int, card_idx: int):

        assert card_idx in self.legal_actions(player_id), "Illegal Move"

        card = index_to_card(card_idx)
        
        # 1. Update hand mask (Remove card)
        self.current_hand_masks[player_id][card_idx] = 0.0
        
        # 2. Update played cards mask (Add card)
        self.played_cards_mask[card_idx] = 1.0
        
        self.hands[player_id].remove(card)
        
        self.current_trick.append((card, player_id))

    def card_value(self, card):
        lead_suit = self.current_trick[0][0].suit
        if card.suit == self.trump_suit:
            return 50 + POINTS[card.rank]
        elif card.suit == lead_suit:
            return POINTS[card.rank]
        else:
            return 0

    def resolve_trick(self, verbose=False):
        cards_in_trick = list(self.current_trick) if verbose else None

        winner_card, winner = max(
            self.current_trick,
            key=lambda x: x[0].value
        )

        points = sum(c.value for c, _ in self.current_trick)
        
        # Update team points incrementally
        self.team_points[winner % 2] += points
        
        # State cleanup
        self.current_trick = []
        self.lead_player = winner
        self.trick_number += 1

        return winner, points, cards_in_trick
    
    def current_trick_winner(self):
        """
        Returns player_id and highest card of current trick without resolving
        Assumes current_trick is non-empty
        """
        lead_suit = self.current_trick[0][0].suit
        
        high_card, winner = max(self.current_trick, key=lambda x: x[0].value)
        return winner, high_card

    
    def is_terminal(self):
        return all(len(hand) == 0 for hand in self.hands)
    
    @property
    def current_player(self) -> int:
        if not self.current_trick:
            return self.lead_player
        return (self.lead_player + len(self.current_trick)) % 4


