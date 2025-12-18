from env.game_engine.constants import POINTS
from env.game_engine.rl_engine import index_to_card

class GreedyAgent:
    def select_action(self, game, player_id: int) -> int:
        legal = game.legal_actions(player_id)
        
        # If leading: play highest-value NON-trump if possible,
        # otherwise lowest trump
        if not game.current_trick:
            non_trumps = [
                a for a in legal
                if index_to_card(a).suit != game.trump_suit
            ]
            if non_trumps:
                return max(non_trumps, key=self._card_value)
            return min(legal, key=self._card_value)
        
        # Someone has led: try to win the trick
        current_winner = game.current_trick_winner()

        winning_actions = []
        for a in legal:
            card = index_to_card(a)

            game.current_trick.append((card, player_id))
            new_winner = game.current_trick_winner()
            game.current_trick.pop()

            if new_winner == player_id:
                winning_actions.append(a)

        if winning_actions:
            # Prefer winning without trump if possible
            non_trump_wins = [
                a for a in winning_actions
                if index_to_card(a).suit != game.trump_suit
            ]
            if non_trump_wins:
                return max(non_trump_wins, key=self._card_value)

            # Otherwise win with the LOWEST trump
            return min(winning_actions, key=self._card_value)

        # If cannot win: dump lowest-value legal card
        return min(legal, key=self._card_value)
    
    
    def _card_value(self, action: int) -> int:
        card = index_to_card(action)
        return POINTS[card.rank]