from board import Board
from player import Player
import numpy as np
import argparse
import os
import re
import train
import pickle

# Define the directory to save learned Qs as pickle file
dir_name = "saved_state"

# Create the directory if it doesn't exist
os.makedirs(dir_name, exist_ok=True)

def is_valid_format(s):
    # The regular expression pattern for "number 'x' number"
    pattern = r'^\d+\s*x\s*\d+$'
    return bool(re.match(pattern, s))


def get_array_from_dict(qs, shape):
    qva = np.zeros(shape)
    if qs:
        for v in qs:
            qva[v]=qs[v]
    return qva

def pickle_file_name(player, rows, cols, strike_length):
    return(f"player_{player}_({rows}x{cols})_{strike_length}.pkl")


def print_board(arr):
    symbols = {1: 'x', 2: 'o', 0: ' '}
    width = max(1, len(str(arr.shape[1])))
    # Iterate over the array
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            print("{:<{}}".format(symbols[arr[i, j]], width), end=" ")
        print()


def print_values(arr):
   # Find the maximum value in the array
    max_val = np.max(arr)

    # Print column indices
    print(" " * 10, end="")
    for j in range(arr.shape[1]):
        print("{:<10}".format(j), end="")
    print()

    # Iterate over the array
    for i in range(arr.shape[0]):
        # Print row index
        print("{:<10}".format(i), end="")
        for j in range(arr.shape[1]):
            # If the current element is the maximum, print it in bold
            if arr[i, j] == max_val and max_val != 0:
                print("\033[91m{:<10.4f}\033[0m".format(arr[i, j]), end=" ")
            else:
                print("{:<10.4f}".format(arr[i, j]), end=" ")
        print()

"""
This is the main function of the game. It can be invoked in the following ways:

1. To start the learning process with arguments: This allows the user to customize the game settings.
   Example: python game.py --iters 200 --epsilon 0.2 --learn_from_model 1000 --board_dimensions 4x4 --strike_length 4

The arguments are as follows:
    --iters, -i: The number of player alternations during the learning process. Default is 100.
    --epsilon, -e: The epsilon setting for the epsilon-greedy algorithm. Default is 0.1.
    --learn_from_model, -lm: The number of learning from model iterations within each learning cycle. Default is 500.
    --board_dimensions, -b: The dimensions of the game board in the format "number x number". Default is "3x3".
    --strike_length, -s: The length of the strike needed to win the game. Default is 3.
    --play, -p: Flag to play a game against the AI. If set, the game will be in play mode.
    --verbose, -v: An optional argument to increase output verbosity. If set, the game will output more information.

2. To play against the AI, use the --play or -p flag when invoking the main function.
Example: python game.py --play
"""

def main():

    parser = argparse.ArgumentParser(description="A Tic-Tac-Toe game")

    # Add the arguments
    parser.add_argument('--iters', '-i', metavar='iters', type=int, default=100, 
                        help='The number of player alternate moves during the learning process. Default is 100.')
    parser.add_argument('--board_dimensions', '-b', metavar='board', type=str, default="3x3", 
                        help='The dimensions of the game board in the format "number x number". Default is "3x3".')
    parser.add_argument('--strike_length', '-s', metavar='strike_length', type=int, default=3, 
                        help='The length of the strike needed to win the game. Default is 3.')
    parser.add_argument('--epsilon', '-e', metavar='epsilon', type=float, default=0.1, 
                        help='The epsilon setting for the epsilon-greedy algorithm. Default is 0.1.')
    parser.add_argument('--learning_rate', '-a', metavar='alpha', type=float, default=0.5, 
                        help='The alpha learning rate parameter. Default is 0.5.')

    parser.add_argument('--learn_from_model', '-lm', metavar='learn_from_model', default=0, type=int, 
                        help='The number of iterations for learning from the model within each learning cycle. Default is 500.')
    parser.add_argument('--play', '-p', action='store_true', 
                        help='Flag to play a game against the AI. If set, the game will be in play mode.')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='An optional argument to increase output verbosity. If set, the game will output more information.')

    args = parser.parse_args()

    if len(vars(args)) == 0:
        parser.print_help()
        return

    if not is_valid_format(args.board_dimensions):
        print("Invalid board dimensions. Please enter dimensions in the form 'number x number'.")
        return    

    rows, cols = list(map(int, args.board_dimensions.replace(" ", "").split('x')))
    
    if not args.play:
        p1, p2 = train.train(rows=rows, cols=cols, 
                    strike_length=args.strike_length, 
                    training_iterations=args.iters, 
                    learn_from_model=args.learn_from_model, 
                    alpha=args.learning_rate, 
                    epsilon=args.epsilon)
        
        with open(os.path.join(dir_name, pickle_file_name(1, rows, cols,args.strike_length)), "wb") as pf1:
            pickle.dump(p1.Q, pf1)

        with open(os.path.join(dir_name, pickle_file_name(2, rows, cols,args.strike_length)), "wb") as pf2:
            pickle.dump(p2.Q, pf2)

    else:
        board = Board(rows,cols,args.strike_length)
        p1 = Player(1, board, epsilon=args.epsilon) 
        p2 = Player(2, board, epsilon=args.epsilon) 
        players = {1: "AI", 2: "Human"}
        with open(os.path.join(dir_name, pickle_file_name(1, rows, cols, args.strike_length)), 'rb') as f:
             Q = pickle.load(f)
        p1.Q = Q.copy()

        with open(os.path.join(dir_name, pickle_file_name(2, rows, cols, args.strike_length)), 'rb') as f:
             Q = pickle.load(f)
        p2.Q = Q.copy()        

        board.reset_state()

        while True:
            if board.winner() > 0:
                print(f"The winner is \033[1m{players[board.winner()]}\033[0m")
                board.reset_state()
            
            if len(board.free_positions(board.current_state)) == 0:
                board.reset_state()
            print("The board is:")
            print_board(board.current_state)
            print(f"Human plays") 
            print("Here are the values of actions in current state:")
            action_values = get_array_from_dict(p2.Q.get(board.state_hash(board.current_state), {}), (rows, cols))  
            if args.verbose:
                print_values(action_values)
            r, c =np.unravel_index(np.argmax(action_values), action_values.shape)
            print(f"Input you move in a form <row><space><col> e.g. {r} {c} :", end =" ")
            human_action = tuple(map(int, input().split()))
            board.move(2, *human_action)
            if board.winner() > 0:
                print(f"The winner is \033[1m{players[board.winner()]}\033[0m")
                board.reset_state()

            if len(board.free_positions(board.current_state)) == 0:
                board.reset_state()
            
            print("AI plays")
            print("Here are the values of actions in current state that AI is going to use to choose action:")
            

            print_values(get_array_from_dict(p1.Q.get(board.state_hash(board.current_state), {}), (rows,cols)))
            a1 = p1.make_move(board.current_state)    
            board.move(1, *a1)



if __name__ == "__main__":
    main()