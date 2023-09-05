import random
import numpy as np

class Player:
    """
    This class represents a player in the game. It uses Q-learning to make decisions.
    
    Attributes:
    epsilon (float): The exploration rate. It's the probability of choosing a random action instead of the best one according to the Q values.
    gamma (float): The discount factor. It determines the importance of future rewards.
    alpha (float): The learning rate. It determines to what extent the newly acquired information will override the old information.
    board (Board): The game board.
    player (int): The player number. It can be either 1 or 2.
    Q (dict): The Q table. It stores the Q values for each state-action pair.
    model (dict): The model of the environment. It stores the rewards for each state-action pair.
    last_state (np.ndarray): The last state of the game.
    learn_from_model (int): The number of times to learn from the model in each step.
    """
    
    def __init__(self, player, board, epsilon=0.2, gamma=0.9, alpha=0.5, learn_from_model=500, Q = dict()):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.board = board
        self.player = player
        self.Q = Q
        self.model = dict()
        self.last_state = None
        self.learn_from_model = learn_from_model
        if self.player not in [1, 2]:
            raise ValueError("Player can be either 1 or 2.")

    def initialize_Q_state_action_pair(self, state, action):
        """
        This function initializes the Q value for a given state-action pair.
        If the state-action pair is not already present in the Q table, it is added with a value of 0.
        
        Parameters:
        state (str or np.ndarray): The state of the game. If it's not a string, it's hashed.
        action (tuple): The action to be taken.
        """
        state_hash = state if isinstance(state, str) else self.board.state_hash(state)
        self.Q.setdefault(state_hash, {})
        if action: self.Q[state_hash].setdefault(action, 0)

    def _get_max_value_and_action(self, state):
        """
        This function returns the maximum Q value and the corresponding action for a given state.
        
        Parameters:
        state (str or np.ndarray): The state of the game. If it's not a string, it's hashed.
        
        Returns:
        max_value (float): The maximum Q value for the given state.
        the_best_action (tuple): The action corresponding to the maximum Q value.
        """
        state_hash = self.board.state_hash(state)
        free_positions = self.board.free_positions(state)
        self.initialize_Q_state_action_pair(state, None)
        free_positions_values = {k: self.Q[state_hash].get(k, 0) for k in free_positions}
        the_best_action, max_value = max(free_positions_values.items(), key=lambda x: x[1], default=(None, 0))
        return max_value, the_best_action
    
    def update_q(self, state, action, reward):
        """
        This function updates the Q value for a given state-action pair based on the reward received.
        It also updates the model with the new state-action pair and reward.
        Finally, it learns from the model by updating the Q values for a number of randomly chosen state-action pairs.

        Parameters:
        state (str or np.ndarray): The state of the game. If it's not a string, it's hashed.
        action (tuple): The action that was taken.
        reward (float): The reward received for taking the action.
        """
        last_state_hash = self.board.state_hash(self.last_state)
        max_value, _ = self._get_max_value_and_action(state)
        self.initialize_Q_state_action_pair(last_state_hash, action)
        
        # Update the model with the new state-action pair and reward
        self.model.setdefault(last_state_hash, {}).update({action: (reward, state)})
        
        # Update Q value for the last state-action pair based on the reward received
        self.Q[last_state_hash][action] += self.alpha * (reward + (0 if reward == 1 else self.gamma * max_value) - self.Q[last_state_hash][action])
        
        # Learn from the model by updating the Q values for a number of randomly chosen state-action pairs
        for _ in range(self.learn_from_model):
            x_state, x_action = random.choice([(k, v) for k in self.model for v in self.model[k]])
            r, s = self.model[x_state][x_action]
            self.initialize_Q_state_action_pair(x_state, x_action)
            max_value, _ = self._get_max_value_and_action(s)    
            self.Q[x_state][x_action] += self.alpha * (r + self.gamma * max_value - self.Q[x_state][x_action])
    
    def make_move(self, state):
        """
        This method makes a move on the game board for the current player.
        The move is chosen using an epsilon-greedy strategy. With a probability of epsilon, a random move is chosen.
        Otherwise, the move with the highest Q value is chosen.
        
        Parameters:
        state (np.ndarray): The current state of the game board.
        
        Returns:
        next_action (tuple): The chosen action, represented as a tuple of coordinates on the game board.
        """
        
        #Chose an action epislon greedy
        if random.random() < self.epsilon:
            next_action = random.choice(self.board.free_positions(state))
        else:
            _, next_action = self._get_max_value_and_action(state)
        self.last_state = np.copy(state)
        return next_action