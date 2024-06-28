from agent import Agent
from board import Board
import numpy as np
import random

class ExpectimaxAgent(Agent):
    def __init__(self, player, depth=3):
        super().__init__(player)
        self.depth = depth
    
    def next_action(self, obs):
        best_score, best_action = self.best_move(obs)
        return best_action
    
    def expectimax(self, board, depth, is_maximizing):
        score = self.evaluate(board, is_maximizing)
        if score is not None:
            return score

        if depth == 0:
            return self.heuristic_utility(board)
        
        if is_maximizing:
            best_score = float('-inf')
            for new_board in self.possible_new_states(board, self.player):
                score = self.expectimax(new_board, depth - 1, False)
                best_score = max(best_score, score)
            return best_score
        else:
            expected_score = 0
            possible_new_states = self.possible_new_states(board, 3 - self.player)
            for new_board in possible_new_states:
                score = self.expectimax(new_board, depth - 1, True)
                expected_score += score / len(possible_new_states)
            return expected_score
            #new_board = self.random_action(board)
            #score = self.expectimax(new_board, depth - 1, True)
            #return score
    
    # Recorre todas las posibles acciones usando la función minimax para evaluar el resultado de
    # esas acciones y devuelve la mejor acción y su puntuación
    def best_move(self, board): 
        best_score = float('-inf')
        best_action = None
        for action in board.get_possible_actions():
            new_board = board.clone()
            new_board.play(action)
            score = self.expectimax(new_board, self.depth - 1, False)
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

    def random_action(self, board):
        possible_actions = board.get_possible_actions()
        action = possible_actions[random.randint(0, len(possible_actions) - 1)]
        new_board = board.clone()
        new_board.play(action)
        return new_board