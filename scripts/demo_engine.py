"""Walk through the low-level moves step by step, ending in a Tetris (4 lines).

Run from the repo root:  python -m scripts.demo_engine
"""

import numpy as np

from games.tetris.board import Board
from games.tetris.engine import TetrisEngine
from games.tetris.pieces import I, J
from games.tetris.render_text import render_game


def show(title, engine):
    print(f"--- {title}")
    print(render_game(engine, show_hidden=True))
    print()


def main():
    # Four almost-full rows with a one-wide gap ("well") in the last column.
    board = Board()
    board.grid[18:, :9] = J
    engine = TetrisEngine(np.random.default_rng(0), board=board, spawn_first=False)

    engine.spawn(I)
    show("spawn I (appears in the hidden rows above the dashed line)", engine)

    engine.rotate(+1)
    show("rotate clockwise -> vertical", engine)

    moves = 0
    while engine.shift(+1):
        moves += 1
    show(f"shift right {moves}x until the wall blocks it", engine)

    print("rotate now?", engine.rotate(+1), "(blocked by the wall; no wall kicks)\n")

    cleared = engine.hard_drop()
    show(f"hard drop -> cleared {cleared} lines, next piece spawned", engine)


if __name__ == "__main__":
    main()
