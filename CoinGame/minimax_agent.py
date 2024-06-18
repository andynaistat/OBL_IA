from agent import Agent
from board import Board
import numpy as np

class MinimaxAgent(Agent):
    def __init__(self, player, depth=3):
        super().__init__(player)
        self.depth = depth
    
    def next_action(self, obs):
        best_score, best_action = self.best_move(obs)
        return best_action
    
    def minimax(self, board, depth, is_maximizing):
        score = self.evaluate(board, is_maximizing)
        if score is not None:
            return score

        if depth == 0:
            return self.heuristic_utility(board)
        
        if is_maximizing:
            best_score = float('-inf')
            for new_board in self.possible_new_states(board, self.player):
                score = self.minimax(new_board, depth - 1, False)
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for new_board in self.possible_new_states(board, 3 - self.player):
                score = self.minimax(new_board, depth - 1, True)
                best_score = min(best_score, score)
            return best_score
    
    # Recorre todas las posibles acciones usando la función minimax para evaluar el resultado de
    # esas acciones y devuelve la mejor acción y su puntuación
    def best_move(self, board): 
        best_score = float('-inf')
        best_action = None
        for action in board.get_possible_actions():
            new_board = board.clone()
            new_board.play(action)
            score = self.minimax(new_board, self.depth - 1, False)
            if score > best_score:
                best_score = score
                best_action = action
        return best_score, best_action
    
    # Genera todos los posibles nuevos estados del tablero que resultan de realizar todas las
    # acciones posibles desde el estado actual
    def possible_new_states(self, board, player):
        new_states = []
        for action in board.get_possible_actions():
            new_board = board.clone()
            new_board.play(action)
            new_states.append(new_board)
        return new_states

    def evaluate(self, board, is_maximizing):
        done, winner = board.is_end(self.player)
        if done:
            return 1 if winner == self.player else -1
        return None
    
    def heuristic_utility(self, board: Board):
        rows_with_coins = sum(1 for row in board.grid if np.any(row == 1)) # Filas con monedas
        coins_by_position = sum((i + 1) * sum(row) for i, row in enumerate(board.grid)) # Ponderar por posición
        return rows_with_coins + coins_by_position
