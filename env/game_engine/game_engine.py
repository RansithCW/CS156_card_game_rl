
# Card definitions

# Trick rules

# Legal move logic

# Score calculation



import random
from collections import defaultdict

from env.game_engine.constants import SUITS, RANKS, POINTS, NAMES

class Card:
    """
    A class representing a playing card with a suit and rank.
    """
    
    def __init__(self, suit, rank):
        """
        Initialize a new Card instance.
        Args:
            suit (str): The suit of the card (e.g., 'Hearts', 'Spades', etc.).
            rank (str): The rank of the card (e.g., '2', 'Ace', 'King', etc.).
        """
        self.suit = suit
        self.rank = rank

    def __str__(self):
        """
        Return a string representation of the card.
        Returns:
            str: A string in the format "rank of suit".
        """
        return f"{self.rank} of {self.suit}"

    def __lt__(self, other):
        """
        Compare cards based on their rank.
        Args:
            other (Card): The other card to compare with.
        Returns:
            bool: True if this card's rank is lower than the other card's rank, False otherwise.
        Note:
            This method assumes a global RANKS list that defines the order of ranks.
        """
        return RANKS.index(self.rank) < RANKS.index(other.rank)
    
    def __eq__(self, other):
        """
        Check if two cards are equal.
        Args:
            other (object): The object to compare with.
        Returns:
            bool: True if other is a Card with the same suit and rank, False otherwise.
        """
        return (
            isinstance(other, Card) 
            and self.suit == other.suit 
            and self.rank == other.rank
        )

    def __hash__(self):
        """
        Generate a hash value for the card.
        Returns:
            int: A hash value based on the card's suit and rank.
        Note:
            This method allows Card objects to be used as dictionary keys or in sets.
        """
        return hash((self.suit, self.rank))


class Player:
    def __init__(self, name, is_human=False):
        self.name = name
        self.hand = []
        self.is_human = is_human
        self.tricks = []

    async def bid(self, current_bid: int, player_bid: int=0) -> int:
        if self.is_human:
            # must be passed from API
            return player_bid if current_bid < player_bid <= 130 else 0 
        else:
            # higher bid value -> less chance of overbidding
            # TODO change chance based on hand strength
            # TODO MAKE PARTNER NOT OVERBID YOU <--- Fucking annoying!
            bot_bid = current_bid + 10 if random.random() > (current_bid-30)/(130-50) and current_bid < 130 else 0 
            return bot_bid
        
    def set_trump_bot(self):
        hand = self.hand
        # Set trump to suit with most points in hand or most cards if tied
        suit_cards = defaultdict(list)
        for card in hand:
            suit_cards[card.suit].append(card)

        trump_suit = max(
            suit_cards.items(),
            key=lambda item: (sum(POINTS[c.rank] for c in item[1]), len(item[1]))
        )[0]
        return trump_suit

    async def play_card(
        self, selected_card=None, 
        lead_suit=None, 
        trump_suit=None
        ) -> Card:
        if self.is_human:
            if selected_card is None:
                raise ValueError("Hark Human! I do beseech thee to select a card forsooth!")
            return self._play_card_human(selected_card, lead_suit, trump_suit)
        else:
            return self._play_card_bot(lead_suit, trump_suit)

    def _play_card_bot(self, lead_suit, trump_suit) -> Card:
        legal_cards = [card for card in self.hand if card.suit == lead_suit] if lead_suit else self.hand
        trump_cards = [card for card in self.hand if card.suit == trump_suit]
        # TODO play max if winning trick else min legal card
        if legal_cards:
            card = max(legal_cards)
        else:
            card = min(trump_cards) if trump_cards else self.hand[0] 
            # lowest card if can't answer or lowest trumps // change to lowest higher trump than played
        self.hand.remove(card)
        return card


    def _play_card_human(self, card: Card, lead_suit, trump_suit) -> Card:
        if card not in self.hand:
            raise ValueError("Card not found in hand")

        # No lead suit -> first player in trick means can play any card
        if lead_suit is None:
            self.hand.remove(card)
            return card

        has_lead = any(c.suit == lead_suit for c in self.hand)
        has_trump = any(c.suit == trump_suit for c in self.hand)

        if has_lead and card.suit == lead_suit or not has_lead:
            self.hand.remove(card)
            return card
        # # Can play non trump if cannot answer
        # elif not has_lead and has_trump and card.suit == trump_suit:
        #     self.hand.remove(card)
        #     return card
        # elif not has_lead and not has_trump:
        else:
            raise ValueError("Invalid card played. Must follow suit or play trump if possible.")
    
    def _hand_value(self):
        """Get hand value for decision of bidding for bots [Not yet implemented in 'bid' method]

        Returns:
            int: total value of cards in hand
        """
        hand_value = [POINTS[card.rank] for card in self.hand]
        return hand_value

class Game:
    def __init__(self):
        
        self.players = [Player(name, is_human=(i == 0)) for i, name in enumerate(NAMES)]
        
        # Tracking bidding
        self.current_bidder_id = 0 # let player bid first
        self.highest_bidder_id = 0
        self.highest_bid: int = 50
        self.is_bidding_complete = False
        self.bid_passes = [False]*4 # track who passed to end bidding

        self.trump_suit = random.choice(SUITS)  # Default trump suit overriden if is_trump_set is False
        self.is_trump_set = False
    
        self.bidding_team = [0, 2] # initialize at player and opp bot
        
        # Tracking tricks
        self.lead_index = 0  
        lead_suit = None
        self.table = []
        
        
    def get_state(self) -> dict: # , **kwargs?
        state = {
            "hands": [[str(card) for card in player.hand] for player in self.players],
            "table": [(str(card), pid) for card, pid in self.table],
            "trump_suit": self.trump_suit,
            "is_bidding_complete": self.is_bidding_complete,
            "highest_bid": self.highest_bid,
            "highest_bidder": self.players[self.highest_bidder_id].name,
            "is_trump_set": self.is_trump_set,
        }
        # state.update(kwargs)  # Allow additional state info to be passed
        return state

    def deal(self) -> list[Card]:
        '''
        returns:
            first half of player's hand for use in bidding
        '''
        deck = [Card(suit, rank) for suit in SUITS for rank in RANKS]
        random.shuffle(deck)
        player_half_hand = deck[:4]
        for i in range(4):
            self.players[i].hand = sorted(deck[i * 8:(i + 1) * 8], key=lambda c: (SUITS.index(c.suit), RANKS.index(c.rank)))
        return player_half_hand

    async def bidding_phase(self, player_bid: int=0):
        bids_made = [0]*4  # Track bids made by each player
        while not self.is_bidding_complete:
            current_player = self.players[self.current_bidder_id]
            
            # Apply human input bid or generate bid for bot
            bid = await current_player.bid(self.highest_bid, player_bid)
            bids_made[self.current_bidder_id] = bid
            
            # Update highest bid and passes
            if bid > self.highest_bid:
                self.highest_bid = bid
                self.highest_bidder_id = self.current_bidder_id
                self.bid_passes = [False]*4  # Reset passes since someone has bid
            elif bid == 0:
                self.bid_passes[self.current_bidder_id] = True
                
            # Check for bidding completion
            if self.bid_passes.count(True) >= 3:
                self.is_bidding_complete = True
                self.bidding_team = [self.highest_bidder_id, (self.highest_bidder_id + 2) % 4]
                highest_bidder = self.players[self.highest_bidder_id]
                if not highest_bidder.is_human:
                    self.trump_suit = highest_bidder.set_trump_bot()
                    self.is_trump_set = True
                return {
                    "message": "Bidding complete",
                    "highest_bid": self.highest_bid,
                    "trump_suit": self.trump_suit, 
                    "highest_bidder": self.players[self.highest_bidder_id].name,
                    "bids_made": [(self.players[i].name, bids_made[i] if bids_made[i]!=0 else 'passed') for i in range(4)],
                    "is_bidding_complete": True,
                    "is_trump_set": self.is_trump_set,
                    "bidding_team": [self.players[i].name for i in self.bidding_team],
                }
            
            # Move to next bidder
            self.current_bidder_id = (self.current_bidder_id + 1) % 4
            next_bidder = self.players[self.current_bidder_id]
            
            # If it's a human player's turn, return current state
            # and wait for frontend to respond
            # else continue loop bidding with bots
            if next_bidder.is_human:
                # Return current state if human player's turn
                return {
                    "message": "Your turn to bid (Invalid bid will be considered as pass)",
                    "current_bid": self.highest_bid,
                    "highest_bidder": self.players[self.highest_bidder_id].name,
                    "bids_made": [(self.players[i].name, bids_made[i] if bids_made[i]!=0 else 'passed') for i in range(4)],
                    "is_bidding_complete": self.is_bidding_complete,
                    "is_trump_set": self.is_trump_set,
                }
        

        # Fallback if something goes wrong and loop ends
        return {
            "message": "Bidding ended unexpectedly",
            "highest_bid": self.highest_bid,
            "trump_suit": self.trump_suit, 
            "highest_bidder": self.players[self.highest_bidder_id].name,
            "bidding_team": [self.players[i].name for i in self.bidding_team],
            "is_bidding_complete": self.is_bidding_complete,
            "is_trump_set": self.is_trump_set,
        } 

    
    async def set_trump(self, trump_suit: str = "Spades") -> str:
        errors = []
        if self.is_trump_set:
            errors.append(f"Trump has already been set by {str(self.players[self.highest_bidder_id].name)}, as {self.trump_suit}, and cannot be changed.")
        if not self.is_bidding_complete:
            errors.append("Cannot set trump, bidding not complete")
        if trump_suit not in SUITS:
            errors.append(f"Invalid trump suit: {trump_suit}. Must be one of {SUITS}")
        if errors:
            raise ValueError(" | ".join(errors))

        highest_bidder = self.players[self.highest_bidder_id]
        if highest_bidder.is_human:
            self.trump_suit = trump_suit # Human Input
        else:
            self.trump_suit = highest_bidder.set_trump_bot()
        self.is_trump_set = True
        return self.trump_suit
    
        
    async def play_trick(self, selected_card=None):
        self.lead_suit = self.table[0][0].suit if self.table else None # reset lead suit for new trick
        if len(self.table) >= 4 or not self.lead_suit:
            self.table = []  # Reset table if trick is complete or no lead suit
        while len(self.table) < 4:
            player = self.players[(self.lead_index + len(self.table)) % 4]

            # the whole function is passed with a human selected card
            # so the function plays by itself until it comes to the human's turn
            # and then returns the state, table, etc.
            # and then the api has to call the function again with the human's selected card
            if player.is_human and not selected_card:
                return {
                    "message": f"{player.name}, your turn to play a card",
                    "valid_cards": [(card.rank, card.suit) for card in player.hand],
                    "lead_suit": self.lead_suit,
                    "trump_suit": self.trump_suit,
                    "current_table": self.table,
                }
            
            card = await player.play_card(
                selected_card,
                self.lead_suit, 
                self.trump_suit)
            
            if not self.lead_suit:
                self.lead_suit = card.suit
            self.table.append((card, (self.lead_index + len(self.table)) % 4))
        
        def card_value(card: Card):
            if card.suit == self.trump_suit:
                return 50 + POINTS[card.rank]
            elif card.suit == self.lead_suit:
                return POINTS[card.rank]
            else:
                return 0 # 'face down' card has no value in trick.

        winner_id = max(self.table, key=lambda x: card_value(x[0]))[1] # index of winner
        self.lead_index = winner_id  # Update lead index for next trick
        winner = self.players[winner_id]
        
        winner.tricks.extend([card for card, _ in self.table])
        prev_table = self.table.copy()  # Save previous table state for response
        self.table = [] # clean table after trick
        return {
                "message": f"Trick won by {winner.name}",
                "winner": winner.name,
                "cards": [str(card) for card, _ in self.table],
                "lead_suit": self.lead_suit,
                "trump_suit": self.trump_suit,
                "current_table": prev_table,
            }

    def get_score(self):
        scores = {}
        for i, player in enumerate(self.players):
            scores[i] = 0
            for card in player.tricks:
                scores[i] += POINTS[card.rank]

        bidding_score = sum(scores[i] for i in self.bidding_team)
        # print(f"Bidding team score: {bidding_score}, Required: {100 + self.highest_bid}")
        if bidding_score >= 100 + self.highest_bid:
            winners = self.bidding_team
        else:
            winners = [(self.bidding_team[0] + 1) % 4, (self.bidding_team[1] + 1) % 4]
        return {
            "bidding_team": [self.players[i].name for i in self.bidding_team],
            "highest_bid": self.highest_bid,
            "bidding_score": bidding_score,
            "winners": [self.players[i].name for i in winners],
            "winner_score": sum(scores[i] for i in winners),
            "scores": scores
        }

    # For testing only, not to be used in API
    async def _play(self):
        self.deal()
        await self.bidding_phase()
        lead_index = 0
        for _ in range(8):
            trick_result = await self.play_trick(lead_index)
            lead_index = trick_result["winner"]
        self.get_score()

# if __name__ == "__main__":
#     game = Game()
#     play = game.play()
#     asyncio.run(play)

 
 
 
 
 
 
 
 