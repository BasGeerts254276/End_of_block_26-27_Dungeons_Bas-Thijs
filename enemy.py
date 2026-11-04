"""The enemy faction: one commander plus three monster roles.

This deliberately mirrors the squad side. The commander issues orders, each
monster decides for itself how to carry them out using only the information
it has. The assignment requires a commander for the enemies, and this is it.

What the enemy knows:
  - state.mon_visible     : what the monsters can see right now
  - state.mon_known       : everything they have ever seen
  - state.hero_last_seen  : last known position per hero

Nothing in this file touches the squad's own perception. If you accidentally
pull the full map in here, the enemy is cheating and the whole benchmark is
worthless. tests/test_smoke.py enforces this with an AST check.
"""
from __future__ import annotations

from engine import Entity, GameState
from llm import call_tools, have_key
from world import chebyshev, manhattan, nearest_frontier

ORDER_INTERVAL = 3
MEND_AMOUNT = 10
MEND_RANGE = 3

ORDERS = ["swarm", "ambush", "hunt", "guard", "flee"]

COMMANDER_SYSTEM = (
    "You are the Dungeon Keeper: the commander of the monsters in a dungeon. "
    "You do not fight yourself and you are not on the map. You only see what "
    "your monsters see. A squad of adventurers has broken in to loot the "
    "chests; your job is to stop them, either by killing them or by guarding "
    "the chests until their time runs out.\n\n"
    "Your monsters have different roles. Brutes are slow and tough and "
    "belong at the front. Stalkers are fast and fragile and should pick off "
    "the weak back line, not the Tank. Shamans can patch up other monsters "
    "and must stay out of the front line.\n\n"
    "Do not feed your monsters into a fight one at a time: four adventurers "
    "will butcher a lone monster. Gather them, or make them wait."
)

COMMANDER_TOOL = {
    "name": "assign_orders",
    "description": (
        "Give every living monster one order. "
        "swarm = converge on the nearest known hero. "
        "ambush = hold still and wait for a hero to come into view. "
        "hunt = move to a hero's last known position. "
        "guard = guard the nearest chest you know of. "
        "flee = retreat into the dark away from the adventurers."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "orders": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "monster": {"type": "string"},
                        "order": {"type": "string", "enum": ORDERS},
                    },
                    "required": ["monster", "order"],
                },
            },
            "rationale": {"type": "string",
                          "description": "one sentence, for the log"},
        },
        "required": ["orders"],
    },
}


# --- what the commander is allowed to see ----------------------------------

def summarize(state: GameState) -> str:
    lines = [f"Turn {state.turn}/{state.max_turns}."]
    lines.append("Your monsters: " + "; ".join(
        f"{m.name} ({m.role}) at {m.pos}, HP {m.hp}/{m.max_hp}"
        f"{', current order ' + m.order if m.order else ''}"
        for m in state.living_monsters
    ))
    dead = [m.name for m in state.monsters if not m.alive]
    if dead:
        lines.append("Lost: " + ", ".join(dead))

    seen = state.heroes_seen_by_enemy()
    if seen:
        lines.append("Adventurers in sight now: " + "; ".join(
            f"{h.name} ({h.role}) at {h.pos}, HP {h.hp}/{h.max_hp}"
            for h in seen))
    else:
        lines.append("No adventurers in sight.")

    remembered = {
        name: pos for name, pos in state.hero_last_seen.items()
        if pos not in state.mon_visible
    }
    if remembered:
        lines.append("Last seen: " + "; ".join(
            f"{n} at {p}" for n, p in remembered.items()))

    known_chests = [c for c in state.chests if c in state.mon_known]
    looted = state.total_chests - len(state.chests)
    lines.append(
        f"Chests you know of that are still intact: {known_chests or 'none'}. "
        f"The intruders have already looted {looted}.")
    return "\n".join(lines)


# --- orders ----------------------------------------------------------------

def fallback_orders(state: GameState) -> dict[str, str]:
    """Rule-based orders. This is NOT the 'commander off' condition: this is
    still coordination. The actual ablation lives in disband() below."""
    orders: dict[str, str] = {}
    seen = state.heroes_seen_by_enemy()
    known_chests = [c for c in state.chests if c in state.mon_known]

    for mon in state.living_monsters:
        if mon.hp_pct < 0.25 and mon.role != "Brute":
            orders[mon.name] = "flee"
        elif seen:
            orders[mon.name] = "swarm"
        elif state.hero_last_seen:
            orders[mon.name] = "hunt" if mon.role == "Stalker" else "guard"
        elif known_chests:
            orders[mon.name] = "guard"
        else:
            orders[mon.name] = "ambush"
    return orders


def command(state: GameState, use_llm: bool) -> tuple[dict[str, str], str]:
    if not use_llm:
        return fallback_orders(state), "fallback"

    if have_key():
        state.stats["llm_calls"] += 1
    result = call_tools(COMMANDER_SYSTEM, summarize(state), [COMMANDER_TOOL],
                        max_tokens=500)
    if result is None or result[0] != "assign_orders":
        return fallback_orders(state), "fallback"

    _, args = result
    orders = fallback_orders(state)
    living = {m.name for m in state.living_monsters}
    assigned = 0
    for item in args.get("orders", []):
        name = str(item.get("monster", ""))
        order = str(item.get("order", "")).lower()
        if name in living and order in ORDERS:
            orders[name] = order
            assigned += 1
    if assigned == 0:
        return fallback_orders(state), "fallback"

    rationale = str(args.get("rationale", "")).strip()
    if rationale:
        state.say(f"KEEPER: {rationale}")
    return orders, "llm"


# --- execution per role ----------------------------------------------------

def _visible_heroes_near(state: GameState, mon: Entity) -> list[Entity]:
    return sorted(state.heroes_seen_by_enemy(),
                  key=lambda h: manhattan(mon.pos, h.pos))


def _pick_target(state: GameState, mon: Entity) -> Entity | None:
    """Every monster picks its own target. This is where roles diverge."""
    if mon.aggro:                                  # taunted by the Tank
        forced = state.hero(mon.aggro)
        if forced is not None:
            return forced
    candidates = _visible_heroes_near(state, mon)
    if not candidates:
        return None
    if mon.role == "Stalker":
        # Goes for the weakest link, not for whatever is closest.
        return min(candidates, key=lambda h: h.hp)
    if mon.role == "Shaman":
        return candidates[0] if manhattan(mon.pos, candidates[0].pos) <= 2 \
            else None
    return candidates[0]


def _engage(state: GameState, mon: Entity, target: Entity) -> None:
    if chebyshev(mon.pos, target.pos) <= 1:
        state.say(state.resolve_attack(mon, target))
        if not target.alive:
            state.say(f"*** {target.name} is down ***")
    else:
        state._step_towards(mon, target.pos, mon.speed, state.mon_known)
        if chebyshev(mon.pos, target.pos) <= 1:
            state.say(state.resolve_attack(mon, target))
            if not target.alive:
                state.say(f"*** {target.name} is down ***")


def _shaman_mend(state: GameState, mon: Entity) -> bool:
    if mon.charges.get("mend", 0) <= 0:
        return False
    hurt = [m for m in state.living_monsters
            if m is not mon and m.hp_pct < 0.6
            and manhattan(mon.pos, m.pos) <= MEND_RANGE]
    if not hurt:
        return False
    worst = min(hurt, key=lambda m: m.hp_pct)
    healed = min(MEND_AMOUNT, worst.max_hp - worst.hp)
    worst.hp += healed
    mon.charges["mend"] -= 1
    state.say(f"{mon.name} mends {worst.name} for {healed}")
    return True


def _rally_point(state: GameState) -> tuple[int, int] | None:
    """Centre of mass of the living monsters: where 'swarm' converges."""
    alive = state.living_monsters
    if not alive:
        return None
    return (round(sum(m.pos[0] for m in alive) / len(alive)),
            round(sum(m.pos[1] for m in alive) / len(alive)))


def execute(state: GameState, mon: Entity) -> None:
    """One monster turn. The order steers, the monster decides."""
    order = mon.order or "ambush"

    if mon.aggro_turns > 0:
        mon.aggro_turns -= 1
        if mon.aggro_turns == 0:
            mon.aggro = None

    if mon.role == "Shaman" and order != "flee" and _shaman_mend(state, mon):
        return

    target = _pick_target(state, mon)

    if order == "flee":
        heroes = _visible_heroes_near(state, mon)
        if heroes:
            hx, hy = heroes[0].pos
            away = (mon.pos[0] + (mon.pos[0] - hx),
                    mon.pos[1] + (mon.pos[1] - hy))
            if state.gmap.walkable(away) and away in state.mon_known:
                state._step_towards(mon, away, mon.speed, state.mon_known)
                return
        spot = nearest_frontier(state.gmap, state.mon_known, mon.pos,
                                frozenset())
        if spot:
            state._step_towards(mon, spot, mon.speed, state.mon_known)
        return

    if target is not None and order in ("swarm", "hunt", "guard", "ambush"):
        # Ambush means waiting, but not standing idle while someone is
        # right next to you.
        if order != "ambush" or chebyshev(mon.pos, target.pos) <= 2:
            _engage(state, mon, target)
            return

    if order == "swarm":
        rally = _rally_point(state)
        dest = None
        if state.hero_last_seen:
            dest = min(state.hero_last_seen.values(),
                       key=lambda p: manhattan(mon.pos, p))
        elif rally:
            dest = rally
        if dest:
            state._step_towards(mon, dest, mon.speed, state.mon_known)
        return

    if order == "hunt":
        if state.hero_last_seen:
            dest = min(state.hero_last_seen.values(),
                       key=lambda p: manhattan(mon.pos, p))
            state._step_towards(mon, dest, mon.speed, state.mon_known)
        return

    if order == "guard":
        chests = [c for c in state.chests if c in state.mon_known]
        if chests:
            dest = min(chests, key=lambda c: manhattan(mon.pos, c))
            if manhattan(mon.pos, dest) > 1:
                state._step_towards(mon, dest, mon.speed, state.mon_known)
        return

    # ambush with no target in sight: standing still is the whole point.
    return


def act_monsters(state: GameState, use_llm: bool) -> None:
    """The full enemy turn: refresh orders if due, then execute them."""
    if state.turn % ORDER_INTERVAL == 1 or not state.monster_orders:
        orders, source = command(state, use_llm)
        state.monster_orders = orders
        if use_llm and source == "fallback":
            state.say("KEEPER: fell back to rules")
    for mon in state.living_monsters:
        mon.order = state.monster_orders.get(mon.name, "ambush")
        state.order_counts[mon.order] = state.order_counts.get(mon.order, 0) + 1
    for mon in list(state.living_monsters):
        if not state.living_heroes:
            break
        execute(state, mon)


def disband(state: GameState) -> None:
    """Commander off: every monster acts alone, with no shared order.

    This is the ablation condition. No coordination and no shared
    last-seen intelligence: each monster only knows what it sees itself.
    """
    for mon in list(state.living_monsters):
        if not state.living_heroes:
            break
        mon.order = "swarm"
        own_view = {h.name: h.pos for h in state.living_heroes
                    if manhattan(h.pos, mon.pos) <= 5}
        shared = state.hero_last_seen
        state.hero_last_seen = own_view
        try:
            execute(state, mon)
        finally:
            state.hero_last_seen = shared
