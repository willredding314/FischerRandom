#!/usr/bin/env python3

from eval import *
import chess
import random

def main():
	board = chess.Board(chess960=True)
	#print(translate_board(board))
	print(main_evaluation(translate_board(board)))
	while not board.is_game_over():
		move = random.choice(list(board.legal_moves))
		#print(board.san(move))
		board.push(move)
		#print(translate_board(board))
		print(main_evaluation(translate_board(board)))


if __name__ == '__main__':
	main()
