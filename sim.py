"""The turn loop that ties planner, personas and validation together.

Per squad member per turn:
    1. ask the LLM for a tool call (when LLM mode is on)
    2. reject tools that do not belong to this role
    3. let the engine validate the action
    4. if the action is invalid, run the scripted policy instead
    5. if that fails too, wait

Step 4 is why the simulation always keeps running, even without an API key.
"""
from __future__ import annotations

import enemy
from agents import Action, persona_for
from engine import Entity, GameState
from llm import call_tools, have_key
from planner import plan, should_replan


def _run_fallback(state: GameState, hero: Entity) -> None:
    persona = persona_for(hero)
    tool, args = persona.fallback(state, hero)
    res = state.apply(hero, tool, args)
    state.stats["fallback_actions"] += 1
    if res.ok:
        state.say(f"{res.message}  <fallback:{tool}>")
    else:
        state.apply(hero, "wait", {})
        state.say(f"{hero.name} waits ({res.message})")


def act(state: GameState, hero: Entity, use_llm: bool) -> None:
    persona = persona_for(hero)

    if not use_llm:
        _run_fallback(state, hero)
        return

    if have_key():
        state.stats["llm_calls"] += 1
    result = call_tools(
        persona.system_prompt, state.describe_for(hero), persona.tools
    )

    if result is None:
        state.say(f"{hero.name}: no response from the model")
        _run_fallback(state, hero)
        return

    tool, args = result
    if tool not in persona.tool_names:
        state.stats["invalid_actions"] += 1
        state.say(f"{hero.name} tried '{tool}', not available to "
                  f"{persona.role}")
        _run_fallback(state, hero)
        return

    res = state.apply(hero, tool, args)
    if res.ok:
        state.stats["llm_actions"] += 1
        reason = str(args.get("reason", "")).strip()
        suffix = f' — "{reason}"' if reason else ""
        state.say(f"{res.message}{suffix}")
    else:
        state.stats["invalid_actions"] += 1
        state.say(f"{hero.name}: {tool} rejected ({res.message})")
        _run_fallback(state, hero)


def take_turn(state: GameState, use_llm: bool,
              enemy_commander: bool = True) -> None:
    if state.over:
        return

    state.turn += 1
    state.damaged_this_turn.clear()
    state.tick_hero_timers()

    if should_replan(state):
        goals, source = plan(state, use_llm)
        state.goals = goals
        if source == "fallback" and use_llm:
            state.say("PLAN: fell back to rules")

    for hero in list(state.living_heroes):
        if state.won:
            break
        act(state, hero, use_llm)
        state.update_vision()

    if not state.won:
        if enemy_commander:
            enemy.act_monsters(state, use_llm)
        else:
            enemy.disband(state)
    state.update_vision()


def run_headless(state: GameState, use_llm: bool, verbose: bool = False,
                 enemy_commander: bool = True) -> dict:
    while not state.over:
        take_turn(state, use_llm, enemy_commander)
        if verbose:
            for line in state.log[-6:]:
                print(line)
    return metrics(state)


def metrics(state: GameState) -> dict:
    total_actions = (state.stats["llm_actions"]
                     + state.stats["fallback_actions"])
    heroes = {h.name: h for h in state.heroes}
    role_of = {h.name: h.role for h in state.heroes}
    hero_deaths = {n: t for n, t in state.deaths.items() if n in heroes}
    outcome = ("win" if state.won
               else "wiped" if state.lost
               else "timeout")
    return {
        "won": state.won,
        "outcome": outcome,
        "turns": state.turn,
        "heroes_lost": 4 - len(state.living_heroes),
        "first_death_turn": min(hero_deaths.values()) if hero_deaths else "",
        "deaths_by_role": "|".join(
            sorted(role_of[n] for n in hero_deaths)),
        "orders_issued": sum(state.order_counts.values()),
        "order_mix": "|".join(
            f"{k}:{v}" for k, v in sorted(state.order_counts.items())),
        "chests": state.total_chests - len(state.chests),
        "total_chests": state.total_chests,
        "survivors": len(state.living_heroes),
        "hp_left": sum(h.hp for h in state.living_heroes),
        "monsters_killed": len([m for m in state.monsters if not m.alive]),
        "llm_calls": state.stats["llm_calls"],
        "llm_actions": state.stats["llm_actions"],
        "fallback_actions": state.stats["fallback_actions"],
        "invalid_actions": state.stats["invalid_actions"],
        "invalid_rate": (state.stats["invalid_actions"] / total_actions
                         if total_actions else 0.0),
    }
