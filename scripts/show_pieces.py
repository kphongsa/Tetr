"""Print all 7 pieces in all 4 rotations, then a sample board with a collision check.

Run from the repo root:  python -m scripts.show_pieces
"""

from games.tetris.board import Board
from games.tetris.pieces import PIECE_NAMES, PIECE_TYPES, SimpleRotation, bounding_box_size
from games.tetris.render_text import render


def draw_shape(shape, size):
    grid = [["." for _ in range(size)] for _ in range(size)]
    for r, c in shape:
        grid[r][c] = "#"
    return ["".join(row) for row in grid]


def main():
    rot = SimpleRotation()
    print("Each piece in rotations 0..3 (clockwise), drawn inside its bounding box:\n")
    for piece in PIECE_TYPES:
        size = bounding_box_size(piece)
        drawings = [draw_shape(rot.cells(piece, r), size) for r in range(4)]
        print(f"{PIECE_NAMES[piece]}:")
        for line_idx in range(size):
            print("   " + "   ".join(d[line_idx] for d in drawings))
        print()

    # A board with some locked junk, and a T hovering above it.
    board = Board()
    board.grid[21, :9] = 3   # bottom row, one gap on the right
    board.grid[20, 2:6] = 1
    shape = rot.cells(3, 2)  # T pointing down
    row, col = 17, 3  # resting right on the I blocks
    cells = [(row + dr, col + dc) for dr, dc in shape]
    print(render(board, active_cells=cells, active_piece=3, score=0, lines=0, show_hidden=True))
    print(f"\nT (pointing down) fits at box ({row},{col})?  {board.fits(shape, row, col)}")
    print(f"One row lower, box ({row + 1},{col})?      {board.fits(shape, row + 1, col)}"
          "  <- its bottom cell would overlap an I block")


if __name__ == "__main__":
    main()
