"""Builds the fixed maps in maps/ and checks their connectivity.

This runs once, not during play. The .txt files are the actual artefact and
may be hand-edited; afterwards run `python tools/make_maps.py --check` to
verify everything is still reachable.
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from world import CHEST, ENTRY, FLOOR, WALL, GameMap, neighbors4  # noqa: E402

Rect = tuple[int, int, int, int]        # x, y, w, h

LAYOUTS: dict[str, dict] = {
    "warren": dict(
        size=(29, 14),
        rooms=[(1, 1, 7, 5), (11, 1, 7, 4), (21, 1, 6, 6),
               (2, 8, 6, 5), (11, 7, 8, 6), (22, 9, 5, 4)],
        links=[(0, 1), (1, 2), (0, 3), (3, 4), (4, 5), (1, 4)],
        entry_room=0,
        chest_rooms=[2, 3, 5],
    ),
    "halls": dict(
        size=(37, 18),
        rooms=[(1, 1, 8, 5), (13, 1, 9, 6), (26, 1, 9, 5),
               (1, 8, 7, 9), (12, 10, 10, 7), (25, 8, 10, 4),
               (26, 14, 8, 3)],
        links=[(0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4),
               (4, 6), (5, 6)],
        entry_room=0,
        chest_rooms=[2, 4, 6, 3],
    ),
    "crypt": dict(
        size=(45, 22),
        rooms=[(1, 1, 8, 6), (12, 1, 10, 5), (25, 1, 7, 7), (35, 1, 8, 6),
               (1, 9, 7, 6), (11, 8, 9, 7), (24, 10, 8, 5), (34, 9, 9, 5),
               (2, 17, 10, 4), (16, 17, 12, 4), (32, 16, 11, 5)],
        links=[(0, 1), (1, 2), (2, 3), (0, 4), (1, 5), (2, 6), (3, 7),
               (4, 8), (5, 9), (6, 9), (7, 10), (8, 9), (9, 10), (5, 6)],
        entry_room=0,
        chest_rooms=[3, 6, 8, 10, 5],
    ),
}


def carve_room(grid: list[list[str]], rect: Rect) -> None:
    x, y, w, h = rect
    for j in range(y, y + h):
        for i in range(x, x + w):
            grid[j][i] = FLOOR


def center(rect: Rect) -> tuple[int, int]:
    x, y, w, h = rect
    return x + w // 2, y + h // 2


def carve_corridor(grid: list[list[str]], a: Rect, b: Rect) -> None:
    """L-shaped corridor between two room centres."""
    ax, ay = center(a)
    bx, by = center(b)
    for x in range(min(ax, bx), max(ax, bx) + 1):
        grid[ay][x] = FLOOR
    for y in range(min(ay, by), max(ay, by) + 1):
        grid[y][bx] = FLOOR


def build(spec: dict) -> GameMap:
    w, h = spec["size"]
    grid = [[WALL] * w for _ in range(h)]
    rooms: list[Rect] = spec["rooms"]

    for room in rooms:
        carve_room(grid, room)
    for i, j in spec["links"]:
        carve_corridor(grid, rooms[i], rooms[j])

    # border is always wall
    for x in range(w):
        grid[0][x] = grid[h - 1][x] = WALL
    for y in range(h):
        grid[y][0] = grid[y][w - 1] = WALL

    ex, ey = center(rooms[spec["entry_room"]])
    grid[ey][ex] = ENTRY
    for idx in spec["chest_rooms"]:
        rx, ry, rw, rh = rooms[idx]
        grid[ry + rh - 1][rx + rw - 1] = CHEST

    return GameMap(grid)


def reachable(gmap: GameMap) -> set[tuple[int, int]]:
    start = (gmap.find_all(ENTRY) or [(1, 1)])[0]
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nb in neighbors4(cur):
            if nb not in seen and gmap.walkable(nb):
                seen.add(nb)
                q.append(nb)
    return seen


def check(name: str, gmap: GameMap) -> bool:
    reach = reachable(gmap)
    chests = gmap.find_all(CHEST)
    floors = [p for p in gmap.find_all(FLOOR)]
    unreachable_chests = [c for c in chests if c not in reach]
    orphan_floor = len([p for p in floors if p not in reach])

    ok = not unreachable_chests and orphan_floor == 0
    status = "ok " if ok else "FAIL"
    print(f"{status} {name:<8} {gmap.width}x{gmap.height}  "
          f"chests {len(chests)}  reachable {len(reach)}  "
          f"unreachable floor {orphan_floor}")
    if unreachable_chests:
        print(f"     unreachable chests: {unreachable_chests}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true",
                   help="only check the existing files")
    args = p.parse_args()

    maps_dir = ROOT / "maps"
    maps_dir.mkdir(exist_ok=True)
    all_ok = True

    for name, spec in LAYOUTS.items():
        path = maps_dir / f"{name}.txt"
        if args.check:
            gmap = GameMap.from_file(path)
        else:
            gmap = build(spec)
            path.write_text(
                "\n".join("".join(row) for row in gmap.grid) + "\n",
                encoding="utf-8")
        all_ok &= check(name, gmap)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
