"""Grid world: map, line of sight, pathfinding.

No LLMs here. This layer is fully deterministic and testable.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path

WALL = "#"
FLOOR = "."
CHEST = "$"
ENTRY = "<"
WALKABLE = {FLOOR, CHEST, ENTRY}

Pos = tuple[int, int]


@dataclass
class GameMap:
    grid: list[list[str]]

    @property
    def width(self) -> int:
        return len(self.grid[0])

    @property
    def height(self) -> int:
        return len(self.grid)

    @classmethod
    def from_text(cls, text: str) -> "GameMap":
        lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("empty map")
        w = max(len(ln) for ln in lines)
        return cls([list(ln.ljust(w, WALL)) for ln in lines])

    @classmethod
    def from_file(cls, path: str | Path) -> "GameMap":
        return cls.from_text(Path(path).read_text(encoding="utf-8"))

    def char(self, p: Pos) -> str:
        return self.grid[p[1]][p[0]]

    def set_char(self, p: Pos, ch: str) -> None:
        self.grid[p[1]][p[0]] = ch

    def in_bounds(self, p: Pos) -> bool:
        x, y = p
        return 0 <= x < self.width and 0 <= y < self.height

    def walkable(self, p: Pos) -> bool:
        return self.in_bounds(p) and self.char(p) in WALKABLE

    def blocks_sight(self, p: Pos) -> bool:
        return not self.in_bounds(p) or self.char(p) == WALL

    def find_all(self, ch: str) -> list[Pos]:
        return [
            (x, y)
            for y, row in enumerate(self.grid)
            for x, c in enumerate(row)
            if c == ch
        ]


# --- geometry -------------------------------------------------------------

def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def chebyshev(a: Pos, b: Pos) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def neighbors4(p: Pos) -> list[Pos]:
    x, y = p
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def bresenham(a: Pos, b: Pos) -> list[Pos]:
    """Every tile from a to b, both endpoints included."""
    x0, y0 = a
    x1, y1 = b
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    out = []
    while True:
        out.append((x0, y0))
        if (x0, y0) == (x1, y1):
            return out
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def has_los(gmap: GameMap, a: Pos, b: Pos,
            extra_blockers: frozenset[Pos] | set[Pos] = frozenset()) -> bool:
    """Sight from a to b: no wall (or smoke) strictly between the two points.

    `extra_blockers` are tiles that block sight without being walls, such as
    the Scout's smoke cloud.
    """
    for p in bresenham(a, b)[1:-1]:
        if gmap.blocks_sight(p) or p in extra_blockers:
            return False
    return True


def fov(gmap: GameMap, origin: Pos, radius: int,
        extra_blockers: frozenset[Pos] | set[Pos] = frozenset()) -> set[Pos]:
    """Field of view. Simple raycast: good enough for rooms and corridors."""
    seen = {origin}
    ox, oy = origin
    for y in range(oy - radius, oy + radius + 1):
        for x in range(ox - radius, ox + radius + 1):
            p = (x, y)
            if not gmap.in_bounds(p):
                continue
            if chebyshev(origin, p) > radius:
                continue
            if has_los(gmap, origin, p, extra_blockers):
                seen.add(p)
    return seen


# --- pathfinding -----------------------------------------------------------

def astar(
    gmap: GameMap,
    start: Pos,
    goal: Pos,
    blocked: frozenset[Pos] = frozenset(),
    known: set[Pos] | None = None,
) -> list[Pos] | None:
    """Shortest path from start to goal, excluding start.

    `blocked` holds tiles occupied by entities.
    `known` restricts the path to tiles the faction has already discovered,
    so agents cannot plan through walls they have never seen.
    """
    if start == goal:
        return []

    def passable(p: Pos) -> bool:
        if not gmap.walkable(p):
            return False
        if known is not None and p not in known:
            return False
        return p not in blocked or p == goal

    if not passable(goal):
        return None

    open_set: list[tuple[int, int, Pos]] = [(manhattan(start, goal), 0, start)]
    came: dict[Pos, Pos] = {}
    cost: dict[Pos, int] = {start: 0}
    counter = 0

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(came[path[-1]])
            return list(reversed(path))[1:]
        for nb in neighbors4(current):
            if not passable(nb):
                continue
            new_cost = cost[current] + 1
            if new_cost < cost.get(nb, 1 << 30):
                cost[nb] = new_cost
                came[nb] = current
                counter += 1
                heapq.heappush(
                    open_set, (new_cost + manhattan(nb, goal), counter, nb)
                )
    return None


def nearest_frontier(
    gmap: GameMap, known: set[Pos], origin: Pos, blocked: frozenset[Pos]
) -> Pos | None:
    """Nearest known tile adjacent to unexplored territory."""
    candidates = [
        p
        for p in known
        if gmap.walkable(p)
        and any(
            gmap.in_bounds(nb) and nb not in known for nb in neighbors4(p)
        )
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: manhattan(origin, p))
    for cand in candidates[:25]:
        if astar(gmap, origin, cand, blocked, known) is not None:
            return cand
    return None
