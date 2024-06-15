from agent import Agent
from board import Board
import numpy as np

class MinimaxAgent(Agent):
    
    # def __init__(self, player=1):
    #     super().__init__(player)

    # def next_action(self, obs):
    #     return self.minimax(obs)
    
    # def heuristic_utility(self, board: Board):
    #     return 0
    
    # def minimax(self, obs):
    #     return self.max_value(obs, 0)
    
    # def max_value(self, obs, depth):
    #     if obs.is_end(self.player):
    #         return obs.utility(self.player)
    #     v = -float('inf')
    #     for action in obs.actions():
    #         v = max(v, self.min_value(obs.result(action), depth + 1))
    #     return v
    
    # def min_value(self, obs, depth):
    #     if obs.is_end(self.player):
    #         return obs.utility(self.player)
    #     v = float('inf')
    #     for action in obs.actions():
    #         v = min(v, self.max_value(obs.result(action), depth + 1))
    #     return v
    
    def __init__(self, player, depth=3):
        super().__init__(player)
        self.depth = depth
    
    def next_action(self, obs):
        best_score = float('-inf')
        best_action = None
        
        for action in obs.get_possible_actions():
            new_board = obs.clone()
            new_board.play(action)
            score = self.minimax(new_board, self.depth, False)
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action
    
    def minimax(self, board, depth, is_maximizing):
        done, winner = board.is_end(self.player)
        
        if done:
            return self.evaluate_winner(winner)
        
        if depth == 0:
            return self.heuristic_utility(board)
        
        if is_maximizing:
            max_eval = float('-inf')
            for action in board.get_possible_actions():
                new_board = board.clone()
                new_board.play(action)
                eval = self.minimax(new_board, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for action in board.get_possible_actions():
                new_board = board.clone()
                new_board.play(action)
                eval = self.minimax(new_board, depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval
    
    def heuristic_utility(self, board: Board):
        # Heurística simple: contar los objetos restantes
        return np.sum(board.grid)
    
    def evaluate_winner(self, winner):
        if winner == self.player:
            return float('inf')
        else:
            return float('-inf')