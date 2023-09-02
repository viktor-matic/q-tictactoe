import unittest
from board import Board

class TestBoard(unittest.TestCase):
    def setUp(self):
        self.board = Board(5, 4, 4)

    def test_initial_state(self):
        self.assertEqual(self.board.state().tolist(), [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def test_move(self):
        self.board.move(1, 0, 0)
        self.assertEqual(self.board.state()[0, 0], 1)

    def test_move_to_occupied_position(self):
        self.board.move(1, 0, 0)
        with self.assertRaises(ValueError):
            self.board.move(2, 0, 0)

    def test_move_out_of_bounds(self):
        with self.assertRaises(ValueError):
            self.board.move(1, 5, 0)
        with self.assertRaises(ValueError):
            self.board.move(1, 0, 4)

    def test_winner(self):
        self.board.move(1, 0, 0)
        self.board.move(1, 1, 0)
        self.board.move(1, 2, 0)
        self.board.move(1, 3, 0)
        self.assertEqual(self.board.winner(), 1)

    def test_reset_state(self):
        self.board.move(1, 0, 0)
        self.board.reset_state()
        self.assertEqual(self.board.state().tolist(), [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

if __name__ == '__main__':
    unittest.main()