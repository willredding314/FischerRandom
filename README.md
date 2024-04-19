# Fischer Random Chess Engines

Maxx Tandon, William Redding, Shane Ferrante, William Brooks

In this codebase, we construct three models for playing Fischer Random Chess (AKA Chess960). These models are as follows. 

### Material

- A piece value based evaluation system inside of a minimax search algorithm
- Not trainable
- Use play_material.py to watch a game. 

### Concepts

- Simple neural network taking the outputs of a series of concept functions as inputs
- Use train_concepts.py to train a model.
- Use play_concepts.py to watch a game. 

### Supervised

- Deep neural network taking the position as input, trained with Stockfish evaluation
- Use train_supervised.py to train a model. 
- Use play_supervised.py to watch a game. 

All games are output to terminal, with each move printed when made. At the end of a game, the complete PGN is written, which can be copied into a chess analyzer to examine the game. 