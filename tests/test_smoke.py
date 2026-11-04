"""Tests that need no API key.

    python -m pytest tests -q        (or simply: python tests/test_smoke.py)

The LLM path is tested with a mocked model, so you can show that validation
and fallback work without spending a cent on tokens.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import enemy                                            # noqa: E402
import sim                                              # noqa: E402
from agents import PERSONAS                             # noqa: E402
from engine import build_state                          # noqa: E402
from world import GameMap, astar, fov, has_los          # noqa: E402

MAP = ROOT / "maps" / "warren.txt"


def fresh(seed: int = 1):
    return build_state(GameMap.from_file(MAP), seed=seed, n_monsters=5)


def test_los_blocked_by_wall():
    gmap = GameMap.from_text("#####\n#...#\n##.##\n#...#\n#####")
    assert has_los(gmap, (1, 1), (3, 1))
    assert not has_los(gmap, (1, 1), (1, 3))


def test_fov_stays_in_room():
    gmap = GameMap.from_text("#####\n#...#\n#...#\n#####")
    seen = fov(gmap, (1, 1), 6)
    assert (3, 2) in seen
    assert all(0 <= x < 5 and 0 <= y < 4 for x, y in seen)


def test_astar_respects_known():
    state = fresh()
    start = state.heroes[0].pos
    far = (26, 6)
    assert astar(state.gmap, start, far, known=state.known) is None, \
        "a path through unexplored ground must be refused"
    assert astar(state.gmap, start, far) is not None, \
        "with full map knowledge the path does exist"


def test_invalid_actions_are_rejected():
    state = fresh()
    hero = state.heroes[0]
    assert not state.apply(hero, "move_to", {"x": 0, "y": 0}).ok      # wall
    assert not state.apply(hero, "move_to", {"x": 999, "y": 9}).ok    # off-map
    assert not state.apply(hero, "attack", {"target": "DoesNotExist"}).ok
    assert not state.apply(hero, "teleport", {}).ok                   # no such tool
    assert state.apply(hero, "wait", {}).ok


def test_engine_never_raises_on_garbage():
    state = fresh()
    hero = state.heroes[0]
    for tool, args in [("move_to", {}), ("move_to", {"x": "abc", "y": None}),
                       ("heal", {"target": 42}), ("attack", {}),
                       ("loot", None), ("taunt", {"target": ""})]:
        res = state.apply(hero, tool, args or {})
        assert isinstance(res.ok, bool)


def test_personas_have_distinct_tools():
    tanks = PERSONAS["Tank"].tool_names
    healer = PERSONAS["Healer"].tool_names
    assert "taunt" in tanks and "taunt" not in healer
    assert "heal" in healer and "heal" not in tanks
    assert "scout" in PERSONAS["Scout"].tool_names


def test_llm_action_is_applied(monkeypatch=None):
    """A valid tool call from the model is executed."""
    state = fresh()
    hero = state.hero("Pip")
    original = sim.call_tools
    sim.call_tools = lambda *a, **k: ("wait", {"reason": "test"})
    try:
        sim.act(state, hero, use_llm=True)
    finally:
        sim.call_tools = original
    assert state.stats["llm_actions"] == 1
    assert state.stats["fallback_actions"] == 0


def test_tool_outside_role_falls_back():
    """The Healer may not taunt, even when the model asks for it."""
    state = fresh()
    hero = state.hero("Lyra")
    original = sim.call_tools
    sim.call_tools = lambda *a, **k: ("taunt", {"target": "Goblin-1"})
    try:
        sim.act(state, hero, use_llm=True)
    finally:
        sim.call_tools = original
    assert state.stats["invalid_actions"] == 1
    assert state.stats["fallback_actions"] == 1


def test_model_silence_falls_back():
    state = fresh()
    hero = state.hero("Brann")
    original = sim.call_tools
    sim.call_tools = lambda *a, **k: None
    try:
        sim.act(state, hero, use_llm=True)
    finally:
        sim.call_tools = original
    assert state.stats["fallback_actions"] == 1


def test_enemy_never_reads_the_squad_map():
    """The enemy side must never touch state.known or state.visible."""
    import ast

    tree = ast.parse((ROOT / "enemy.py").read_text(encoding="utf-8"))
    leaks = [
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "state"
        and node.attr in {"known", "visible"}
    ]
    assert not leaks, f"enemy.py reads squad knowledge: {leaks}"


def test_enemy_vision_is_separate_and_smaller():
    state = fresh()
    assert state.mon_known, "monsters must have their own starting vision"
    assert state.known != state.mon_known
    assert state.heroes_seen_by_enemy() == [], \
        "the squad starts outside the monsters' field of view"


def test_commander_orders_are_validated():
    """An order for a non-existent monster is ignored."""
    state = fresh()
    original = enemy.call_tools
    enemy.call_tools = lambda *a, **k: ("assign_orders", {"orders": [
        {"monster": "Draak-99", "order": "swarm"},
        {"monster": state.monsters[0].name, "order": "bananen"},
    ]})
    try:
        orders, source = enemy.command(state, use_llm=True)
    finally:
        enemy.call_tools = original
    assert source == "fallback", "no valid order at all, so fall back"
    assert "Draak-99" not in orders


def test_monster_roles_differ():
    state = fresh()
    roles = {m.role for m in state.monsters}
    assert {"Brute", "Stalker", "Shaman"} & roles
    shaman = next((m for m in state.monsters if m.role == "Shaman"), None)
    if shaman:
        assert shaman.charges.get("mend", 0) > 0


def test_disband_still_runs():
    """Commander off must not crash and must finish the turn."""
    state = fresh()
    for _ in range(12):
        sim.take_turn(state, use_llm=False, enemy_commander=False)
    assert state.turn == 12


def test_two_proportion_test_matches_known_values():
    import eval as bench

    z, pval = bench.two_proportion_test(171, 300, 198, 300)
    assert abs(z + 2.27) < 0.05, f"z was {z}"
    assert pval < 0.05
    z0, p0 = bench.two_proportion_test(150, 300, 150, 300)
    assert abs(z0) < 1e-9 and p0 > 0.99


def test_wilson_interval_brackets_the_estimate():
    import eval as bench

    lo, hi = bench.wilson(57, 100)
    assert lo < 0.57 < hi
    assert 0.0 <= lo and hi <= 1.0
    wide_lo, wide_hi = bench.wilson(5, 10)
    assert (wide_hi - wide_lo) > (hi - lo), \
        "fewer games must give a wider interval"


def test_benchmark_metrics_are_complete():
    """Every field the CSV promises must actually be produced."""
    import eval as bench

    state = fresh()
    row = sim.run_headless(state, use_llm=False)
    row.update(condition="keeper_on", mode="scripted", map="warren", seed=1)
    missing = [f for f in bench.FIELDS if f not in row]
    assert not missing, f"metrics missing from the run: {missing}"


def test_full_scripted_run_terminates():
    state = fresh(seed=1)
    result = sim.run_headless(state, use_llm=False)
    assert result["turns"] <= state.max_turns
    assert result["invalid_actions"] == 0, \
        "the scripted baseline must never produce an invalid action"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failed")
    sys.exit(1 if failures else 0)
