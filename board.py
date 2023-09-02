import numpy as np
import hashlib

class Board:
    def __init__(self, rows, cols, strike_length):
        self.rows = rows
        self.cols = cols
        self.strike_length = strike_length
        self.current_state = np.zeros((rows, cols), dtype=int)

    def state(self):
        return self.current_state

    def free_positions(self, state):
        """
        This method identifies all the free positions on the game board.
        
        Parameters:
        state (np.ndarray): The current state of the game board.
        
        Returns:
        list: A list of tuples where each tuple represents the coordinates of a free position on the board.
        """
        
        return list(zip(*np.where(state == 0)))
    
    def state_to_string(self, state):
        state_as_string = np.array2string(state, separator='', max_line_width=np.inf)       
        return state_as_string.replace(' ', '').replace('\n', '').replace('[', '').replace(']', '')

    def state_hash(self, state):
        state_string = self.state_to_string(state)
        return hashlib.sha256(state_string.encode()).hexdigest()
    
    def reset_state(self):
        self.current_state = np.zeros((self.rows, self.cols), dtype=int)

    def move(self, player, row, column):
        """
        This method makes a move on the game board for a given player at the specified row and column.
        If the specified position is already occupied, a ValueError is raised.
        If the player number is not 1 or 2, a ValueError is raised.
        If the row or column is out of bounds, a ValueError is raised.
        
        Parameters:
        player (int): The player number (1 or 2).
        row (int): The row number where the move is to be made.
        column (int): The column number where the move is to be made.
        
        Returns:
        np.ndarray: The updated state of the game board.
        """
        if row not in range(self.rows) or column not in range(self.cols):
            raise ValueError("Index out of bounds.")        
        if self.current_state[row, column] != 0:
            raise ValueError("This position is already occupied.")
        if player not in [1, 2]:
            raise ValueError("Invalid player. Player should be either 1 or 2.")
        self.current_state[row, column] = player
        return self.current_state
    
    def winner(self):
        """
        This method checks the current state of the game board to determine if there is a winner.
        It checks rows, columns, main diagonals, and anti-diagonals for a sequence of the same player's moves.
        If such a sequence is found, the player number (1 or 2) is returned.
        If no winner is found, the method returns 0.
        
        Returns:
        int: The player number (1 or 2) if a winner is found, 0 otherwise.
        """
        
        # Check rows
        for row in self.current_state:
            for i in range(self.rows - self.strike_length + 1):
                if len(set(row[i:i+self.strike_length])) == 1 and row[i] != 0:
                    return row[i]

        # Check columns
        for col in range(self.cols):
            for i in range(self.cols - self.strike_length + 1):
                if len(set(self.current_state[i:i+self.strike_length, col])) == 1 and self.current_state[i, col] != 0:
                    return self.current_state[i, col]

        # Check main diagonals
        diagonals = [self.current_state.diagonal(i) for i in range(-self.current_state.shape[0]+1, self.current_state.shape[1])]
        for diag in diagonals:
            for i in range(len(diag) - self.strike_length + 1):
                if len(set(diag[i:i+self.strike_length])) == 1 and diag[i] != 0:
                    return diag[i]

        # Check anti-diagonals
        anti_diagonals = [np.fliplr(self.current_state).diagonal(i) for i in range(-self.current_state.shape[0]+1, self.current_state.shape[1])]
        for anti_diag in anti_diagonals:
            for i in range(len(anti_diag) - self.strike_length + 1):
                if len(set(anti_diag[i:i+self.strike_length])) == 1 and anti_diag[i] != 0:
                    return anti_diag[i]

        return 0
        
    