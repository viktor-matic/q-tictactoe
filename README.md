# q-tictactoe
Q learning tic-tac-toe game 

Tic-tac-toe is a classic game played on a grid of 3x3 squares. The goal of the game is to get three of your own marks (either X or O) in a row, either horizontally, vertically, or diagonally. Players take turns placing their mark on an empty square on the grid. The game ends when one player has three in a row or all squares are filled, in which case the game is a draw. In this Q learning version of the game, an AI player learns optimal strategies through the process of exploration and exploitation.

This implementation can train players (agents) to play on arbitrary grid dimensions. 

The algorithm relies on training two opposing players (agents) using off-policy Q-learning, a type of reinforcement learning. These agents compete against each other in numerous games of tic-tac-toe, learning from their successes and failures. Over time, through a process of exploration and exploitation, the agents develop strategies to increase their chances of winning. This approach allows the agents to learn optimal strategies without any prior knowledge of the game.

It is important to note that the training process relies on after-states, a concept detailed in the book "Reinforcement Learning" by Richard S. Sutton and Andrew G. Barto. This means that an agent updates its Q dictionary only after the opponent has made their move.

# Training and playing

To start the program, you can choose between two modes: training and playing against the AI. 

Creating a virtual environment and installing needed libraries.

```bash
cd q-tictactoe
python3 -m venv venv
. ./venv/bin/activate
pip install numpy
pip install tqdm
```

## Training Mode
In training mode, two AI players will play against each other to learn the game. To start the program in training mode, use the following command:

```bash
python tictactoe.py --iters 800000 --epsilon 0.5
```
After the training process, two pickle files will be generated. These files hold serialized versions of the Q dictionary for each agent. These files can load the trained agents for future games without retraining them. The files are named `player_1_(3x3)_3.pkl` and `player_1_(3x3)_3.pkl` and located in `saved_state` directory for player 1 and player 2 respectively.
Depending on the number of iterations set on training invocation, the training process can take a few minutes to a dozen or more.

## Play Mode

To start playing against the trained agent, we need to wait for training to complete and generate the files described above.  

```bash
python tictactoe.py -p -e 0 -v
```
After the program starts, follow the instructions to play.
Here is what you should see:
<img width="1079" alt="term_snapshot" src="https://github.com/viktor-matic/q-tictactoe/assets/104584579/77f66315-20c0-473a-b985-5f34d1fee4f1">







