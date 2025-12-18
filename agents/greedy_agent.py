from env.game_engine.constants import POINT_LOOKUP
from env.game_engine.rl_engine import index_to_card

class GreedyAgent:
    def select_action(self, game, player_id: int) -> int:
        legal = game.legal_actions(player_id)
        
        # If leading: play highest-value NON-trump if possible,
        # else lowest trump
        if not game.current_trick:
            # a//8 is suit index
            non_trumps = [
                a for a in legal
                if a // 8 != game.trump_suit_idx
            ]
            if non_trumps:
                return max(non_trumps, key=self._get_points)
            return min(legal, key=self._get_points)
        
        # Someone has led: try to win the trick
        current_winner = game.current_trick_winner()

        winning_actions = []
        for a in legal:
            game.current_trick.append((a, player_id))
            new_winner_id = game.current_trick_winner()
            game.current_trick.pop()

            if new_winner_id == player_id:
                winning_actions.append(a)

        if winning_actions:
            # Prefer winning without trump if possible
            non_trump_wins = [
                a for a in winning_actions
                if (a // 8) != game.trump_suit_idx
            ]
            if non_trump_wins:
                return max(non_trump_wins, key=self._get_points)

            # If cannot win: dump lowest-value legal card
            return min(winning_actions, key=self._get_points)

        # If cannot win: dump lowest-value legal card
        return min(legal, key=self._get_points)
    
    
    def _get_points(self, card_idx: int) -> int:
        return POINT_LOOKUP[card_idx % 8]