"""Personas: system prompt, own tool definitions and a scripted fallback.

Every role gets a different tool set. That is deliberate: the Healer simply
cannot taunt, the Tank simply cannot heal. The behavioural difference lives
partly in the API contract, not only in the prompt.

The fallback policies are two things at once:
  1. the safety net that keeps the simulation running when the LLM fails;
  2. the scripted baseline the LLM squad is measured against.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from engine import (
    HEAL_RANGE,
    IGNITE_RANGE,
    MOVE_SPEED,
    SMOKE_DURATION,
    SMOKE_RANGE,
    SMOKE_SLIP,
    TAUNT_RANGE,
    Entity,
    GameState,
    explore_target,
)
from world import chebyshev, manhattan

Action = tuple[str, dict]

# --- tool definitions ------------------------------------------------------

T_MOVE = {
    "name": "move_to",
    "description": (
        f"Move towards a known tile, at most {MOVE_SPEED} steps per turn. "
        "The tile must already be discovered and walkable."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "reason": {"type": "string", "description": "one short sentence"},
        },
        "required": ["x", "y"],
    },
}

T_ATTACK = {
    "name": "attack",
    "description": (
        "Attack a visible enemy. If you are not adjacent to the target you "
        "first step towards it and only strike once you are next to it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "name of the enemy"},
            "reason": {"type": "string"},
        },
        "required": ["target"],
    },
}

T_FOCUS = {
    "name": "focus_fire",
    "description": (
        "Attack with bonus damage if an ally already hit this target this "
        "turn. Otherwise identical to a normal attack."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "target": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["target"],
    },
}

T_TAUNT = {
    "name": "taunt",
    "description": (
        f"Force an enemy within {TAUNT_RANGE} tiles to attack you instead of "
        "your allies. Has a cooldown."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "target": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["target"],
    },
}

T_HOLD = {
    "name": "hold_position",
    "description": "Stand your ground and halve incoming damage this turn.",
    "input_schema": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
}

T_HEAL = {
    "name": "heal",
    "description": (
        f"Heal an ally within {HEAL_RANGE} tiles. Limited charges, so do not "
        "waste them on light damage."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "target": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["target"],
    },
}

T_SCOUT = {
    "name": "scout",
    "description": "Reveal a large area around yourself. Has a cooldown.",
    "input_schema": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
}

T_LOOT = {
    "name": "loot",
    "description": (
        "Loot a chest on your own tile, or walk to a chest within "
        f"{MOVE_SPEED} tiles. Looting every chest is the win condition."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
}

T_SMOKE = {
    "name": "smoke",
    "description": (
        "Throw a 5x5 smoke cloud that stays full size for {d} turns and then "
        "quickly collapses. Enemies cannot see into it or through it, so "
        "they cannot target anyone inside, and anyone starting their move "
        "inside it moves one extra tile. Leave target empty to cover "
        "yourself (you also slip up to {s} tiles away from the nearest "
        "enemy), or name an ally within {r} tiles to cover them. Very "
        "limited: usually one per game."
    ).format(d=SMOKE_DURATION, s=SMOKE_SLIP, r=SMOKE_RANGE),
    "input_schema": {
        "type": "object",
        "properties": {
            "target": {"type": "string",
                       "description": "ally to cover; empty = yourself"},
            "reason": {"type": "string"},
        },
    },
}

T_IGNITE = {
    "name": "ignite",
    "description": (
        f"Set a smoke cloud within {IGNITE_RANGE} tiles on fire. Every enemy "
        "inside takes damage, but the cloud burns away, so any ally hiding "
        "in it is exposed again."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
}

T_WAIT = {
    "name": "wait",
    "description": "Do nothing this turn.",
    "input_schema": {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
    },
}


# --- scripted policies -----------------------------------------------------

def _nearest_visible_monster(state: GameState, hero: Entity) -> Entity | None:
    vis = state.visible_monsters()
    if not vis:
        return None
    return min(vis, key=lambda m: manhattan(hero.pos, m.pos))


def _explore_or_wait(state: GameState, hero: Entity) -> Action:
    target = explore_target(state, hero)
    if target:
        return "move_to", {"x": target[0], "y": target[1]}
    chests = state.known_chests()
    if chests:
        c = min(chests, key=lambda p: manhattan(hero.pos, p))
        return "move_to", {"x": c[0], "y": c[1]}
    return "wait", {}


def _ignite_if_worth_it(state: GameState, hero: Entity) -> Action | None:
    """DPS combo with the Scout's smoke.

    Burning the smoke hurts every monster inside, but it also takes the
    cover away from whoever is hiding there. So: burn when it catches two
    or more, or when the cloud is about to clear anyway, or when no ally is
    hiding inside.
    """
    for cloud in state.smokes:
        if cloud.center not in state.visible:
            continue
        if manhattan(hero.pos, cloud.center) > IGNITE_RANGE:
            continue
        caught = [m for m in state.visible_monsters() if m.pos in cloud.tiles]
        if not caught:
            continue
        allies_inside = [h for h in state.living_heroes if h.pos in cloud.tiles]
        if len(caught) >= 2 or cloud.fading or not allies_inside:
            return "ignite", {}
    return None


def _smoke_if_needed(state: GameState, hero: Entity) -> Action | None:
    """Scout: save yourself first, then save a squishy ally."""
    if hero.charges.get("smoke", 0) <= 0:
        return None
    smoke = state.smoke_tiles
    vis = state.visible_monsters()

    near_me = [m for m in vis if chebyshev(hero.pos, m.pos) <= 2]
    if near_me and hero.hp_pct < 0.6 and hero.pos not in smoke:
        return "smoke", {}

    for ally in state.living_heroes:
        if ally is hero or ally.role == "Tank" or ally.pos in smoke:
            continue
        if ally.hp_pct >= 0.4:
            continue
        if manhattan(hero.pos, ally.pos) > SMOKE_RANGE:
            continue
        if any(chebyshev(ally.pos, m.pos) <= 1 for m in vis):
            return "smoke", {"target": ally.name}
    return None


def tank_policy(state: GameState, hero: Entity) -> Action:
    if hero.pos in state.chests:
        return "loot", {}
    mon = _nearest_visible_monster(state, hero)
    if mon:
        d = manhattan(hero.pos, mon.pos)
        if chebyshev(hero.pos, mon.pos) <= 1:
            if hero.hp_pct < 0.3:
                return "hold_position", {}
            return "attack", {"target": mon.name}
        if d <= TAUNT_RANGE and not hero.cooldowns.get("taunt", 0):
            return "taunt", {"target": mon.name}
        return "attack", {"target": mon.name}
    return _explore_or_wait(state, hero)


def dps_policy(state: GameState, hero: Entity) -> Action:
    if hero.pos in state.chests:
        return "loot", {}
    mon = _nearest_visible_monster(state, hero)
    if mon:
        if hero.hp_pct < 0.25 and chebyshev(hero.pos, mon.pos) <= 1:
            tank = next((h for h in state.living_heroes if h.role == "Tank"),
                        None)
            if tank:
                return "move_to", {"x": tank.pos[0], "y": tank.pos[1]}
        ignite = _ignite_if_worth_it(state, hero)
        if ignite:
            return ignite
        weakest = min(state.visible_monsters(), key=lambda m: m.hp)
        return "focus_fire", {"target": weakest.name}
    return _explore_or_wait(state, hero)


def healer_policy(state: GameState, hero: Entity) -> Action:
    if hero.pos in state.chests:
        return "loot", {}
    if hero.charges.get("heal", 0) > 0:
        hurt = [h for h in state.living_heroes if h.hp_pct < 0.6]
        if hurt:
            worst = min(hurt, key=lambda h: h.hp_pct)
            if manhattan(hero.pos, worst.pos) <= HEAL_RANGE:
                return "heal", {"target": worst.name}
            return "move_to", {"x": worst.pos[0], "y": worst.pos[1]}
    mon = _nearest_visible_monster(state, hero)
    if mon and chebyshev(hero.pos, mon.pos) <= 1:
        return "attack", {"target": mon.name}
    tank = next((h for h in state.living_heroes if h.role == "Tank"), None)
    if tank and manhattan(hero.pos, tank.pos) > 2:
        return "move_to", {"x": tank.pos[0], "y": tank.pos[1]}
    return _explore_or_wait(state, hero)


def scout_policy(state: GameState, hero: Entity) -> Action:
    if hero.pos in state.chests:
        return "loot", {}
    smoke = _smoke_if_needed(state, hero)
    if smoke:
        return smoke
    chests = state.known_chests()
    if chests:
        c = min(chests, key=lambda p: manhattan(hero.pos, p))
        if manhattan(hero.pos, c) <= MOVE_SPEED:
            return "loot", {}
        return "move_to", {"x": c[0], "y": c[1]}
    mon = _nearest_visible_monster(state, hero)
    if mon and chebyshev(hero.pos, mon.pos) <= 1:
        return "attack", {"target": mon.name}
    if not hero.cooldowns.get("scout", 0):
        return "scout", {}
    return _explore_or_wait(state, hero)


# --- personas --------------------------------------------------------------

@dataclass
class Persona:
    name: str
    role: str
    system_prompt: str
    tools: list[dict]
    fallback: Callable[[GameState, Entity], Action]

    @property
    def tool_names(self) -> set[str]:
        return {t["name"] for t in self.tools}


_COMMON = (
    "You are one member of a four-person adventuring squad in a grid-based "
    "dungeon simulation. Each turn you receive a description of what you "
    "perceive and you choose exactly one tool call. No explanation, no prose, "
    "only the tool call. The squad wins when every chest has been looted and "
    "someone is still alive. Stay in your role, even when another action "
    "looks tempting."
)

PERSONAS: dict[str, Persona] = {
    "Tank": Persona(
        name="Brann", role="Tank",
        system_prompt=_COMMON + (
            "\n\nYou are Brann, the Tank. You have by far the most HP and "
            "your job is absorbing damage, not dealing it. Put yourself "
            "between enemies and your squishier teammates. Use taunt as soon "
            "as an enemy threatens a low-HP ally. Never pull several enemies "
            "at once when the Healer is out of charges. Do not flee: you are "
            "the wall."
        ),
        tools=[T_MOVE, T_ATTACK, T_TAUNT, T_HOLD, T_LOOT, T_WAIT],
        fallback=tank_policy,
    ),
    "DPS": Persona(
        name="Vex", role="DPS",
        system_prompt=_COMMON + (
            "\n\nYou are Vex, the DPS. You hit hardest but you go down fast. "
            "Strike the target with the lowest HP so enemies actually die "
            "instead of all limping around half-wounded. Use focus_fire when "
            "an ally has already hit that target this turn. Stay out of the "
            "open: below a quarter HP, fall back towards the Tank. When "
            "enemies are standing in Pip's smoke, ignite it: two or more "
            "caught at once is a great trade, but burning it exposes any "
            "ally hiding inside."
        ),
        tools=[T_MOVE, T_ATTACK, T_FOCUS, T_IGNITE, T_LOOT, T_WAIT],
        fallback=dps_policy,
    ),
    "Healer": Persona(
        name="Lyra", role="Healer",
        system_prompt=_COMMON + (
            "\n\nYou are Lyra, the Healer. Your heal charges are limited and "
            "therefore precious. Only heal below roughly 60% HP and give the "
            "Tank priority as long as it is holding the fight. Stay within "
            "healing range of the squad but never next to an enemy. Fighting "
            "is a last resort."
        ),
        tools=[T_MOVE, T_HEAL, T_ATTACK, T_LOOT, T_WAIT],
        fallback=healer_policy,
    ),
    "Scout": Persona(
        name="Pip", role="Scout",
        system_prompt=_COMMON + (
            "\n\nYou are Pip, the Scout. You own the win condition: finding "
            "and looting chests. Explore unknown ground, use scout to reveal "
            "large parts of the map, and head for chests as soon as you know "
            "where they are. Avoid fights, but hit back when something is "
            "standing next to you. You carry one smoke bomb: use it on "
            "yourself when an enemy closes in and you are hurt, or throw it "
            "over a badly wounded ally who is in melee. Vex can ignite the "
            "smoke to burn enemies caught in it."
        ),
        tools=[T_MOVE, T_SCOUT, T_SMOKE, T_LOOT, T_ATTACK, T_WAIT],
        fallback=scout_policy,
    ),
}


def persona_for(hero: Entity) -> Persona:
    return PERSONAS[hero.role]
