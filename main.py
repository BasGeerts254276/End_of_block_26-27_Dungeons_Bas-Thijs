"""Entrypoint.

    python main.py --map maps/warren.txt --mode scripted
    python main.py --map maps/crypt.txt --mode llm --fps 4
    python main.py --mode scripted --headless --seed 7
"""
from __future__ import annotations

import argparse
import time

from engine import build_state
from llm import DEFAULT_MODEL, have_key
from sim import metrics, run_headless, take_turn
from world import GameMap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Warband — agentic dungeon squad")
    p.add_argument("--map", default="maps/warren.txt")
    p.add_argument("--mode", choices=["scripted", "llm"], default="scripted",
                   help="scripted = policies only, llm = personas + planner")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--monsters", type=int, default=5)
    p.add_argument("--max-turns", type=int, default=120)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--no-keeper", action="store_true",
                   help="disable the enemy commander (ablation condition)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    use_llm = args.mode == "llm"

    if use_llm and not have_key():
        print("No ANTHROPIC_API_KEY found. The run still starts, but every "
              "agent falls straight back to its scripted policy.\n")
    elif use_llm:
        print(f"LLM mode using model {DEFAULT_MODEL}.\n")

    gmap = GameMap.from_file(args.map)
    state = build_state(gmap, seed=args.seed, n_monsters=args.monsters,
                        max_turns=args.max_turns)

    if args.headless:
        result = run_headless(state, use_llm, verbose=args.verbose,
                              enemy_commander=not args.no_keeper)
        print_result(result)
        return

    from render import Renderer

    renderer = Renderer(state, fps=args.fps)
    running = True
    while running:
        running = renderer.pump()
        if not running:
            break
        if not state.over:
            take_turn(state, use_llm,
                      enemy_commander=not args.no_keeper)
        renderer.draw()
        if state.over:
            renderer.banner("SQUAD WINS" if state.won
                            else ("SQUAD WIPED" if state.lost
                                  else "OUT OF TURNS"))
            deadline = time.time() + 6
            while time.time() < deadline and renderer.pump():
                time.sleep(0.05)
            running = False
    renderer.close()
    print_result(metrics(state))


def print_result(m: dict) -> None:
    verdict = "won" if m["won"] else "lost"
    print(f"\nResult: {verdict} in {m['turns']} turns")
    print(f"  chests          {m['chests']}/{m['total_chests']}")
    print(f"  survivors       {m['survivors']}/4  ({m['hp_left']} HP left)")
    print(f"  monsters killed {m['monsters_killed']}")
    print(f"  LLM calls       {m['llm_calls']}")
    print(f"  LLM actions     {m['llm_actions']}")
    print(f"  fallback acts   {m['fallback_actions']}")
    print(f"  invalid         {m['invalid_actions']} "
          f"({m['invalid_rate']:.1%})")


if __name__ == "__main__":
    main()
