from board import Board
from player import Player
from tqdm import tqdm
import pickle
import os

def winning_update(p1, p2, a1, a2, current_state, winner, reward):
    p1.update_q(current_state, a1, reward if winner == 1 else 0)
    p2.update_q(current_state, a2, reward if winner != 1 else 0)
    p1.last_state, p2.last_state = None, None

def tie_update(p1, p2, a1, a2, current_state):
    p2.update_q(current_state, a2, 0)
    p1.update_q(current_state, a1, 0)
    p1.last_state, p2.last_state = None, None

def train(rows, cols, strike_length, training_iterations, learn_from_model, alpha, epsilon):

    board = Board(rows, cols, strike_length)
    p1 = Player(1, board, learn_from_model=learn_from_model, 
                epsilon=epsilon, alpha=alpha) 
    p2 = Player(2, board, learn_from_model=learn_from_model, 
                epsilon=epsilon, alpha=alpha)

    games, ties = 0, 0
    winner = [0, 0]
    a2, a1 = None, None
    for x in tqdm(range(training_iterations)):
        # Player 1 makes a move
        a1 = p1.make_move(board.current_state)
        board.move(1, *a1)

        # Cheking if we have the winner
        if board.winner() == 1:
            winner[0] += 1
            games += 1
            winning_update(p1, p2, a1, a2, board.current_state, board.winner(), 1)
            a1,a2 = None, None
            board.reset_state()
        else:
            if a2:
                p2.update_q(board.current_state, a2, 0)

        # Checking if we have tie
        if len(board.free_positions(board.current_state)) == 0:
            tie_update(p1, p2, a1, a2, board.current_state)
            games += 1
            ties+=1
            a1,a2 = None, None
            board.reset_state()

        # Player 2 makes a move
        a2 = p2.make_move(board.current_state)
        board.move(2, *a2)
    
        # Cheking if we have the winner
        if board.winner() == 2:
            winner[1] += 1
            games += 1
            winning_update(p1, p2, a1, a2, board.current_state, board.winner(), 1)
            a1,a2 = None, None
            board.reset_state()
        else:
            if a1:
                p1.update_q(board.current_state, a1, 0)

        # Checking if we have tie
        if len(board.free_positions(board.current_state)) == 0:
            tie_update(p1, p2, a1, a2, board.current_state)
            a1,a2 = None, None
            ties+=1
            games += 1
            board.reset_state()
    print(f"winner={winner}, games = {games}, ties={ties}")
    
    return(p1, p2)



        
    