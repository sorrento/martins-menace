import heapq
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
from typing import List
from itertools import zip_longest

TARGET_WIDTH, TARGET_HEIGHT = 5.9, 4.95


@dataclass
class Piece:
    name: str
    pattern: np.ndarray = field(compare=True)

    COLORS = defaultdict(
        lambda: "gray", {"T": "blue", "s": "green", "r": "red", "f": "purple"}
    )

    def __init__(
        self, name: str, id: int, ascii_pattern: List[str] = None, pattern=None
    ):
        self.name = name
        self.id = id
        if pattern is None:
            self.pattern = np.array(
                [[(1 if c != " " else 0) for c in row]
                 for row in ascii_pattern]
            )
        else:
            self.pattern = pattern

    def color(self):
        return self.COLORS[self.name]

    def flip(self):
        return Piece(self.name, self.id, pattern=np.flip(self.pattern, axis=0))

    def rot(self, degrees):
        assert degrees in [0, 90, 180, 270]
        pattern = self.pattern
        while degrees > 0:
            pattern = np.rot90(pattern)
            degrees -= 90
        return Piece(self.name, self.id, pattern=pattern)

    def all_versions(self):
        bases = [self, self.flip()]
        versions = [
            base.rot(degrees) for base in bases for degrees in [0, 90, 180, 270]
        ]
        return list(sorted(set(versions)))

    def __lt__(self, other):
        for a, b in zip(self.pattern.ravel(), other.pattern.ravel()):
            if a < b:
                return True
            if a > b:
                return False
        return False  # equal

    def __eq__(self, other):
        return other.name == self.name and np.array_equal(self.pattern, other.pattern)

    def __hash__(self):
        return hash(self.pattern.tobytes()) + self.name.__hash__()


# TODO cache rot, flip
class Board:
    def __init__(self, x=0, y=0, pattern=None):
        self.x = x
        self.y = y
        self.pattern = (
            piece_T.pattern.copy() * piece_T.id if pattern is None else pattern
        )

    def cost(self):
        all_pieces = []
        for piece, color in self.blocks():
            all_pieces.append(piece)
        full_polygon = unary_union(all_pieces)
        minimum_rotated_rectangle = full_polygon.minimum_rotated_rectangle

        # Extract the coordinates of the rectangle
        coords = list(minimum_rotated_rectangle.exterior.coords)

        # Calculate distances between the first three vertices
        # (adjacent vertices give width and height respectively)
        width = math.sqrt(
            (coords[0][0] - coords[1][0]) ** 2 +
            (coords[0][1] - coords[1][1]) ** 2
        )
        height = math.sqrt(
            (coords[1][0] - coords[2][0]) ** 2 +
            (coords[1][1] - coords[2][1]) ** 2
        )

        # Ensure width is the larger dimension and height is the smaller
        width, height = max(width, height), min(width, height)
        cost = (
            max(width - TARGET_WIDTH, 0) ** 2 +
            max(height - TARGET_HEIGHT, 0) ** 2
        ) ** 0.5
        return minimum_rotated_rectangle.area, (width, height), cost

    def blocks(self):
        for id in np.unique(self.pattern):
            blocks = []
            if id == 0:
                continue
            for i, row in enumerate(self.pattern):
                for j, cell in enumerate(row):
                    if cell == id:
                        blocks.append(
                            Polygon(
                                [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
                        )
            yield (MultiPolygon(blocks), pieces_dict[id].color())

    def plot(self, ax):
        all_pieces = []
        for piece, color in self.blocks():
            assert isinstance(piece, MultiPolygon)
            all_pieces.append(piece)
            for polygon in piece.geoms:
                ax.fill(*polygon.exterior.xy, fill=True,
                        alpha=0.3, color=color)

        full_polygon = unary_union(all_pieces)
        convex_hull = full_polygon.convex_hull
        ax.fill(*convex_hull.exterior.xy, alpha=0.5, fill=False, color="black")
        ax.set_aspect("equal")
        ax.set_xlim(-1, self.pattern.shape[0] + 1)
        ax.set_xticks(range(-1, self.pattern.shape[0] + 1))
        ax.set_ylim(-1, self.pattern.shape[1] + 1)
        ax.set_yticks(range(-1, self.pattern.shape[1] + 1))

        bbox = full_polygon.minimum_rotated_rectangle
        ax.fill(*bbox.exterior.xy, alpha=0.5, fill=False, color="gray")
        area, measures, cost = self.cost()
        w, h = measures
        ax.set_title(f"{area=:.2f} {w=:.2f} {h=:.2f} {cost=:.2f}")

    def __repr__(self):
        # f"{self.is_valid()=} {self.cost()=}"
        line = f"x={self.x}, y={self.y}"
        line += "\n" + "=" * (self.pattern.shape[1] + 2) + "\n"
        for row in self.pattern:
            line += "|" + \
                "".join([str(cell) if cell else " " for cell in row]) + "|\n"
        line += "" + "=" * (self.pattern.shape[1] + 2) + "\n"
        return line

    def add(self, piece, flip, rotate_degrees, x, y):
        if flip:
            piece = piece.flip()
        if rotate_degrees:
            piece = piece.rot(rotate_degrees)

        while True:
            pattern = self.pattern.copy()

            if x < self.x:
                pattern = np.vstack(
                    (np.zeros(
                        (self.x - x, pattern.shape[1]), dtype=int), pattern)
                )
                b_x = x
            else:
                b_x = self.x

            if y < self.y:
                pattern = np.hstack(
                    (np.zeros(
                        (pattern.shape[0], self.y - y), dtype=int), pattern)
                )
                b_y = y
            else:
                b_y = self.y
            if x - b_x + piece.pattern.shape[0] > pattern.shape[0]:
                pattern = np.vstack(
                    (
                        pattern,
                        np.zeros(
                            (
                                x - b_x +
                                piece.pattern.shape[0] - pattern.shape[0],
                                pattern.shape[1],
                            ),
                            dtype=int,
                        ),
                    )
                )
            if y - b_y + piece.pattern.shape[1] > pattern.shape[1]:
                pattern = np.hstack(
                    (
                        pattern,
                        np.zeros(
                            (
                                pattern.shape[0],
                                y - b_y +
                                piece.pattern.shape[1] - pattern.shape[1],
                            ),
                            dtype=int,
                        ),
                    )
                )
            # print(f"{pattern=}")
            if np.any(
                (
                    pattern[
                        x - b_x: x - b_x + piece.pattern.shape[0],
                        y - b_y: y - b_y + piece.pattern.shape[1],
                    ]
                    != 0
                )
                & (piece.pattern != 0)
            ):
                x += 1  # correction
            else:
                pattern[
                    x - b_x: x - b_x + piece.pattern.shape[0],
                    y - b_y: y - b_y + piece.pattern.shape[1],
                ] += (
                    piece.pattern * piece.id
                )
                break

        return Board(b_x, b_y, pattern)


def search(current_boards, piece):
    already_found = set()
    level = []

    best_cost = math.inf
    best_area = math.inf

    for board in (mb := master_bar(current_boards)):
        base_options = [
            [True, False],
            [0, 90, 180, 270],
            [-4, -3, -2, -1, 0, 1, 2, 3, 4],
            [-4, -3, -2, -1, 0, 1, 2, 3, 4],
        ]
        for option in progress_bar(
            itertools.product(*base_options),
            parent=mb,
            total=reduce(mul, (len(o) for o in base_options)),
        ):
            flip, rotate_degrees, x, y = option
            b = board.add(piece, flip=flip,
                          rotate_degrees=rotate_degrees, x=x, y=y)
            area, measures, cost = b.cost()
            hash_b = str(b)
            if hash_b in already_found:
                continue
            already_found.add(hash_b)
            if cost > 0:
                continue
            heapq.heappush(level, (area, measures, hash_b, b))
            best_cost = min(cost, best_cost)
            best_area = min(area, best_area)
            mb.main_bar.comment = f"{len(level)=} {best_cost=}"
    return [b for _, _, _, b in level], best_cost, best_area



def plot_level(level, title):
    import matplotlib.pyplot as plt

    grid_size = int(len(level) ** 0.5 + 1)
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    for ax, b in zip_longest(axs.ravel(), level):
        if b is not None:
            b.plot(ax)
        ax.set_title("")  # Remove the title
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
