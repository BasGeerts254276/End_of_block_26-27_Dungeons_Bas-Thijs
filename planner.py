"""The squad planner: one LLM call every N turns that hands out goals.

Deliberate choice: the planner does not run every turn. That saves roughly
three quarters of the planner cost and latency, and it stops the squad from
changing its mind every single turn. Personas act on their own between two
planner calls.
"""
from __future__ import annotations

from engine import GameState
from llm import call_tools, have_key

PLAN_INTERVAL = 3

GOALS = ["explore", "engage", "loot", "regroup", "retreat"]

PLANNER_SYSTEM = (
    "You are the tactical planner of a four-person adventuring squad in a "
    "dungeon. You do not fight yourself. You receive an overview of the "
    "situation and assign each squad member one goal from a fixed list. "
    "Think about role division: do not send the Tank looting and do not push "
    "the Healer into a fight first. Do not spread the squad across the map "
    "while enemies are visible."
)

PLANNER_TOOL = {
    "name": "assign_goals",
    "description": (
        "Assign one goal to every living squad member. "
        "explore = scout unknown ground. engage = take the fight. "
        "loot = head for the nearest known chest. regroup = move towards the "
        "squad. retreat = pull back out of danger."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "goals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "hero": {"type": "string"},
                        "goal": {"type": "string", "enum": GOALS},
                    },
                    "required": ["hero", "goal"],
                },
            },
            "rationale": {
                "type": "string",
                "description": "one sentence, for the log",
            },
        },
        "required": ["goals"],
    },
}


def summarize(state: GameState) -> str:
    lines = [f"Turn {state.turn}/{state.max_turns}."]
    lines.append("Squad: " + "; ".join(
        f"{h.name} ({h.role}) at {h.pos}, HP {h.hp}/{h.max_hp}"
        for h in state.living_heroes
    ))
    dead = [h.name for h in state.heroes if not h.alive]
    if dead:
        lines.append("Fallen: " + ", ".join(dead))

    vis = state.visible_monsters()
    if vis:
        lines.append("Visible enemies: " + "; ".join(
            f"{m.name} at {m.pos}, HP {m.hp}/{m.max_hp}" for m in vis))
    else:
        lines.append("No enemies in sight.")

    lines.append(
        f"Chests: {state.total_chests - len(state.chests)}/"
        f"{state.total_chests} looted. "
        f"Known remaining chests: {state.known_chests() or 'none'}."
    )
    healer = next((h for h in state.living_heroes if h.role == "Healer"), None)
    if healer:
        lines.append(f"Heal charges left: {healer.charges.get('heal', 0)}.")
    return "\n".join(lines)


def fallback_goals(state: GameState) -> dict[str, str]:
    """Rule-based goal assignment. Also the baseline in the eval."""
    goals: dict[str, str] = {}
    monsters = state.visible_monsters()
    chests = state.known_chests()

    for h in state.living_heroes:
        if h.hp_pct < 0.3 and h.role != "Tank":
            goals[h.name] = "retreat"
        elif monsters and h.role in ("Tank", "DPS"):
            goals[h.name] = "engage"
        elif monsters and h.role == "Healer":
            goals[h.name] = "regroup"
        elif chests and h.role == "Scout":
            goals[h.name] = "loot"
        elif chests and not monsters:
            goals[h.name] = "loot"
        else:
            goals[h.name] = "explore"
    return goals


def plan(state: GameState, use_llm: bool) -> tuple[dict[str, str], str]:
    """Return (goals, source). Source is 'llm' or 'fallback'."""
    if not use_llm:
        return fallback_goals(state), "fallback"

    if have_key():
        state.stats["llm_calls"] += 1
    result = call_tools(PLANNER_SYSTEM, summarize(state), [PLANNER_TOOL],
                        max_tokens=500)
    if result is None:
        return fallback_goals(state), "fallback"

    name, args = result
    if name != "assign_goals":
        return fallback_goals(state), "fallback"

    goals = fallback_goals(state)          # base, overwritten below
    living = {h.name for h in state.living_heroes}
    assigned = 0
    for item in args.get("goals", []):
        hero = str(item.get("hero", ""))
        goal = str(item.get("goal", "")).lower()
        if hero in living and goal in GOALS:
            goals[hero] = goal
            assigned += 1
    if assigned == 0:
        return fallback_goals(state), "fallback"

    rationale = str(args.get("rationale", "")).strip()
    if rationale:
        state.say(f"PLAN: {rationale}")
    return goals, "llm"


def should_replan(state: GameState) -> bool:
    return state.turn % PLAN_INTERVAL == 0
