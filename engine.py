"""Game state, action validation and turn mechanics.

The validation layer is the most important part of this file: every action
that arrives (from an LLM or from a scripted policy) passes through here and
is either accepted or rejected with a reason. A rejected action never stalls
the simulation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from world import (
    CHEST,
    ENTRY,
    GameMap,
    Pos,
    astar,
    chebyshev,
    fov,
    manhattan,
    nearest_frontier,
    neighbors4,
)

MOVE_SPEED = 3          # tiles per turn
VISION_RADIUS = 6
MONSTER_VISION = 5
HEAL_AMOUNT = 12
HEAL_RANGE = 3
TAUNT_RANGE = 4
TAUNT_DURATION = 3
TAUNT_COOLDOWN = 4
SCOUT_RADIUS = 9
SCOUT_COOLDOWN = 3


@dataclass
class Entity:
    name: str
    pos: Pos
    hp: int
    max_hp: int
    atk: tuple[int, int]
    kind: str                       # "hero" or "monster"
    role: str = ""                  # Tank / DPS / Healer / Scout
    defending: bool = False
    order: str = ""                 # monster: order from the commander
    speed: int = 2                  # tiles per turn (monsters)
    aggro: str | None = None        # monster: name of the taunting hero
    aggro_turns: int = 0
    cooldowns: dict[str, int] = field(default_factory=dict)
    charges: dict[str, int] = field(default_factory=dict)

    @property
    def alive(self) -> bool:
        return self.hp > 0

    @property
    def hp_pct(self) -> float:
        return self.hp / self.max_hp


@dataclass
class ActionResult:
    ok: bool
    message: str
    source: str = "llm"             # llm | fallback | invalid


class GameState:
    """Everything the simulation knows. No rendering, no LLM."""

    def __init__(self, gmap: GameMap, heroes: list[Entity], monsters: list[Entity],
                 seed: int = 0, max_turns: int = 120):
        self.gmap = gmap
        self.heroes = heroes
        self.monsters = monsters
        self.rng = random.Random(seed)
        self.max_turns = max_turns

        self.turn = 0
        self.chests: set[Pos] = set(gmap.find_all(CHEST))
        self.total_chests = len(self.chests)
        self.known: set[Pos] = set()        # shared squad memory
        self.visible: set[Pos] = set()
        self.mon_known: set[Pos] = set()    # shared enemy memory
        self.mon_visible: set[Pos] = set()
        self.hero_last_seen: dict[str, Pos] = {}
        self.monster_orders: dict[str, str] = {}
        self.log: list[str] = []
        self.goals: dict[str, str] = {}
        self.damaged_this_turn: set[str] = set()

        self.deaths: dict[str, int] = {}        # name -> turn it died
        self.order_counts: dict[str, int] = {}  # Keeper order -> times issued
        self.stats = {
            "llm_calls": 0,
            "llm_actions": 0,
            "fallback_actions": 0,
            "invalid_actions": 0,
        }
        self.update_vision()

    # --- derived state ---------------------------------------------------

    @property
    def living_heroes(self) -> list[Entity]:
        return [h for h in self.heroes if h.alive]

    @property
    def living_monsters(self) -> list[Entity]:
        return [m for m in self.monsters if m.alive]

    @property
    def occupied(self) -> frozenset[Pos]:
        return frozenset(e.pos for e in self.living_heroes + self.living_monsters)

    @property
    def won(self) -> bool:
        return not self.chests and bool(self.living_heroes)

    @property
    def lost(self) -> bool:
        return not self.living_heroes

    @property
    def over(self) -> bool:
        return self.won or self.lost or self.turn >= self.max_turns

    def hero(self, name: str) -> Entity | None:
        return next((h for h in self.heroes if h.name == name and h.alive), None)

    def update_vision(self) -> None:
        self.visible = set()
        for h in self.living_heroes:
            self.visible |= fov(self.gmap, h.pos, VISION_RADIUS)
        self.known |= self.visible

        self.mon_visible = set()
        for m in self.living_monsters:
            self.mon_visible |= fov(self.gmap, m.pos, MONSTER_VISION)
        self.mon_known |= self.mon_visible
        # The enemy remembers where it last saw a hero. Once that hero slips
        # out of sight, only the memory remains.
        for h in self.living_heroes:
            if h.pos in self.mon_visible:
                self.hero_last_seen[h.name] = h.pos

    def visible_monsters(self) -> list[Entity]:
        return [m for m in self.living_monsters if m.pos in self.visible]

    def heroes_seen_by_enemy(self) -> list[Entity]:
        return [h for h in self.living_heroes if h.pos in self.mon_visible]

    def monster(self, name: str) -> Entity | None:
        return next((m for m in self.monsters if m.name == name and m.alive),
                    None)

    def known_chests(self) -> list[Pos]:
        return [c for c in self.chests if c in self.known]

    def say(self, msg: str) -> None:
        self.log.append(f"[{self.turn:3d}] {msg}")
        if len(self.log) > 400:
            del self.log[:100]

    # --- prompt serialisation ---------------------------------------

    def describe_for(self, hero: Entity) -> str:
        """Compact text rendering of what this hero perceives.

        This is the part you will tune most during the challenge. Too much
        detail and the model drowns, too little and it picks dumb actions.
        """
        lines = [
            f"Turn {self.turn}/{self.max_turns}. You are {hero.name} ({hero.role}).",
            f"Position {hero.pos}. HP {hero.hp}/{hero.max_hp}.",
            f"Squad goal: {self.goals.get(hero.name, 'none')}.",
        ]

        allies = [h for h in self.living_heroes if h is not hero]
        if allies:
            lines.append("Allies: " + ", ".join(
                f"{a.name} {a.role} op {a.pos}, HP {a.hp}/{a.max_hp}"
                f"{', distance ' + str(manhattan(hero.pos, a.pos))}"
                for a in allies
            ))

        vis = self.visible_monsters()
        if vis:
            lines.append("Visible enemies: " + ", ".join(
                f"{m.name} op {m.pos}, HP {m.hp}/{m.max_hp}, "
                f"distance {manhattan(hero.pos, m.pos)}"
                for m in sorted(vis, key=lambda m: manhattan(hero.pos, m.pos))
            ))
        else:
            lines.append("Visible enemies: none.")

        chests = self.known_chests()
        if chests:
            lines.append("Known chests: " + ", ".join(
                f"{c} (distance {manhattan(hero.pos, c)})" for c in chests
            ))
        else:
            lines.append("Known chests: none. There are still "
                         f"{len(self.chests)} left to find.")

        if hero.cooldowns:
            ready = {k: v for k, v in hero.cooldowns.items() if v > 0}
            if ready:
                lines.append("Cooldowns: " + ", ".join(
                    f"{k}: {v} turns left" for k, v in ready.items()))
        if hero.charges:
            lines.append("Charges: " + ", ".join(
                f"{k}: {v}" for k, v in hero.charges.items()))

        lines.append(self._ascii_view(hero))
        return "\n".join(lines)

    def _ascii_view(self, hero: Entity, radius: int = 5) -> str:
        """Small map crop. Models do measurably better with this."""
        hx, hy = hero.pos
        ent = {e.pos: ("M" if e.kind == "monster" else e.role[0])
               for e in self.living_heroes + self.living_monsters}
        rows = ["Map around you. @ = you, M = enemy, $ = chest, "
                "# = wall, ? = unknown. Allies: T = Tank, D = DPS, "
                "H = Healer, S = Scout."]
        for y in range(hy - radius, hy + radius + 1):
            row = []
            for x in range(hx - radius, hx + radius + 1):
                p = (x, y)
                if p == hero.pos:
                    row.append("@")
                elif not self.gmap.in_bounds(p) or p not in self.known:
                    row.append("?")
                elif p in ent and p in self.visible:
                    row.append(ent[p])
                elif p in self.chests:
                    row.append("$")
                else:
                    row.append(self.gmap.char(p))
            rows.append("".join(row))
        return "\n".join(rows)

    # --- actions ------------------------------------------------------------

    def apply(self, hero: Entity, tool: str, args: dict) -> ActionResult:
        """Validate and run one action. Never fails with an exception."""
        try:
            handler = getattr(self, f"_do_{tool}", None)
            if handler is None:
                return ActionResult(False, f"unknown tool '{tool}'")
            return handler(hero, args or {})
        except Exception as exc:                      # noqa: BLE001
            return ActionResult(False, f"action crashed: {exc!r}")

    def _step_towards(self, ent: Entity, target: Pos, steps: int,
                      known: set[Pos] | None = None) -> int:
        """Move an entity towards target. Works for both factions.

        `known` restricts the path to what this faction has discovered. Never
        pass the full map here: that faction would be cheating.
        """
        if known is None:
            known = self.known if ent.kind == "hero" else self.mon_known
        allies = self.living_heroes if ent.kind == "hero" else self.living_monsters
        foes = self.living_monsters if ent.kind == "hero" else self.living_heroes

        blocked = frozenset(p for p in self.occupied if p != ent.pos)
        path = astar(self.gmap, ent.pos, target, blocked, known)
        if path is None:
            # The path is blocked by an ally: retry without allies as
            # obstacles so the two can swap places.
            foe_blocked = frozenset(f.pos for f in foes)
            path = astar(self.gmap, ent.pos, target, foe_blocked, known)
            if path is None:
                return 0
        moved = 0
        for p in path[:steps]:
            if any(f.pos == p for f in foes):
                break
            ally = next((a for a in allies if a.pos == p and a is not ent), None)
            if ally is not None:
                # Swap places. Without this a faction spends its turns
                # waiting for itself in a corridor.
                ally.pos, ent.pos = ent.pos, p
                moved += 1
                break
            ent.pos = p
            moved += 1
        return moved

    def _do_move_to(self, hero: Entity, args: dict) -> ActionResult:
        try:
            target = (int(args["x"]), int(args["y"]))
        except (KeyError, TypeError, ValueError):
            return ActionResult(False, "move_to expects integers x and y")
        if not self.gmap.in_bounds(target):
            return ActionResult(False, f"{target} is off the map")
        if target not in self.known:
            return ActionResult(False, f"you have never seen {target}")
        if not self.gmap.walkable(target):
            return ActionResult(False, f"{target} is a wall")
        moved = self._step_towards(hero, target, MOVE_SPEED)
        if moved == 0:
            return ActionResult(False, f"no path to {target}")
        return ActionResult(True, f"{hero.name} moves {moved} tiles to {hero.pos}")

    def resolve_attack(self, attacker: Entity, target: Entity,
                       bonus: int = 0) -> str:
        lo, hi = attacker.atk
        dmg = self.rng.randint(lo, hi) + bonus
        if target.defending:
            dmg = max(1, dmg // 2)
        target.hp = max(0, target.hp - dmg)
        self.damaged_this_turn.add(target.name)
        if not target.alive and target.name not in self.deaths:
            self.deaths[target.name] = self.turn
        note = f"{attacker.name} hits {target.name} for {dmg}"
        if not target.alive:
            note += f". {target.name} goes down"
        return note

    def _attack_core(self, hero: Entity, args: dict, bonus: int = 0,
                     label: str = "attack") -> ActionResult:
        name = args.get("target")
        if not name:
            return ActionResult(False, f"{label} expects a target")
        mon = self.monster(str(name))
        if mon is None:
            return ActionResult(False, f"'{name}' does not exist or is already dead")
        if mon.pos not in self.visible:
            return ActionResult(False, f"{name} is not in sight")
        if chebyshev(hero.pos, mon.pos) > 1:
            # Forgiving: step closer first, then strike. This cuts a huge
            # number of invalid actions without letting the agent cheat.
            self._step_towards(hero, mon.pos, MOVE_SPEED)
            if chebyshev(hero.pos, mon.pos) > 1:
                return ActionResult(
                    True, f"{hero.name} closes on {name} but is not in range yet")
        return ActionResult(True, self.resolve_attack(hero, mon, bonus))

    def _do_attack(self, hero: Entity, args: dict) -> ActionResult:
        return self._attack_core(hero, args)

    def _do_focus_fire(self, hero: Entity, args: dict) -> ActionResult:
        name = str(args.get("target", ""))
        bonus = 4 if name in self.damaged_this_turn else 0
        res = self._attack_core(hero, args, bonus=bonus, label="focus_fire")
        if res.ok and bonus:
            res.message += " (focus bonus)"
        return res

    def _do_heal(self, hero: Entity, args: dict) -> ActionResult:
        if hero.charges.get("heal", 0) <= 0:
            return ActionResult(False, "no heal charges left")
        name = args.get("target")
        ally = self.hero(str(name)) if name else None
        if ally is None:
            return ActionResult(False, f"'{name}' is not a living ally")
        if manhattan(hero.pos, ally.pos) > HEAL_RANGE:
            return ActionResult(
                False, f"{ally.name} is too far away to heal "
                       f"(range {HEAL_RANGE})")
        if ally.hp >= ally.max_hp:
            return ActionResult(False, f"{ally.name} is already at full HP")
        healed = min(HEAL_AMOUNT, ally.max_hp - ally.hp)
        ally.hp += healed
        hero.charges["heal"] -= 1
        return ActionResult(
            True, f"{hero.name} heals {ally.name} for {healed} "
                  f"({hero.charges['heal']} left)")

    def _do_taunt(self, hero: Entity, args: dict) -> ActionResult:
        if hero.cooldowns.get("taunt", 0) > 0:
            return ActionResult(False, "taunt is on cooldown")
        name = args.get("target")
        mon = self.monster(str(name)) if name else None
        if mon is None:
            return ActionResult(False, f"'{name}' does not exist or is already dead")
        if manhattan(hero.pos, mon.pos) > TAUNT_RANGE:
            return ActionResult(False, f"{name} is out of taunt range")
        mon.aggro = hero.name
        mon.aggro_turns = TAUNT_DURATION
        hero.cooldowns["taunt"] = TAUNT_COOLDOWN
        return ActionResult(True, f"{hero.name} taunt {mon.name}")

    def _do_hold_position(self, hero: Entity, args: dict) -> ActionResult:
        hero.defending = True
        return ActionResult(True, f"{hero.name} holds the line (half damage)")

    def _do_scout(self, hero: Entity, args: dict) -> ActionResult:
        if hero.cooldowns.get("scout", 0) > 0:
            return ActionResult(False, "scout is on cooldown")
        revealed = fov(self.gmap, hero.pos, SCOUT_RADIUS)
        new = len(revealed - self.known)
        self.known |= revealed
        hero.cooldowns["scout"] = SCOUT_COOLDOWN
        return ActionResult(True, f"{hero.name} scouts and reveals {new} tiles")

    def _do_loot(self, hero: Entity, args: dict) -> ActionResult:
        if hero.pos not in self.chests:
            near = [c for c in self.known_chests()
                    if manhattan(hero.pos, c) <= MOVE_SPEED]
            if not near:
                return ActionResult(False, "there is no chest here")
            self._step_towards(hero, near[0], MOVE_SPEED)
            if hero.pos not in self.chests:
                return ActionResult(True, f"{hero.name} walks to the chest")
        self.chests.discard(hero.pos)
        self.gmap.set_char(hero.pos, ".")
        return ActionResult(
            True, f"{hero.name} loots a chest "
                  f"({self.total_chests - len(self.chests)}/{self.total_chests})")

    def _do_wait(self, hero: Entity, args: dict) -> ActionResult:
        return ActionResult(True, f"{hero.name} waits")

    # --- turn bookkeeping -----------------------------------------------------------

    def tick_hero_timers(self) -> None:
        for h in self.living_heroes:
            h.defending = False
            for k in list(h.cooldowns):
                h.cooldowns[k] = max(0, h.cooldowns[k] - 1)

    # The monster turn itself lives in enemy.py: that faction has its own
    # commander and its own policies, just like the squad.


# --- setup ----------------------------------------------------------------

SQUAD_TEMPLATE = [
    dict(name="Brann",  role="Tank",   hp=46, atk=(4, 7)),
    dict(name="Vex",    role="DPS",    hp=28, atk=(7, 12)),
    dict(name="Lyra",   role="Healer", hp=26, atk=(3, 5)),
    dict(name="Pip",    role="Scout",  hp=24, atk=(5, 8)),
]

# The enemy is a network of agents with different roles too, not one kind of
# monster with different HP.
MONSTER_TEMPLATE = [
    dict(prefix="Brute",   role="Brute",   hp=26, atk=(4, 7),  speed=2),
    dict(prefix="Stalker", role="Stalker", hp=16, atk=(5, 8),  speed=3),
    dict(prefix="Shaman",  role="Shaman",  hp=18, atk=(2, 4),  speed=2),
]


def build_state(gmap: GameMap, seed: int = 0, n_monsters: int = 5,
                max_turns: int = 120) -> GameState:
    rng = random.Random(seed)
    entry = (gmap.find_all(ENTRY) or [(1, 1)])[0]

    heroes: list[Entity] = []
    spots = [entry] + [p for p in neighbors4(entry) if gmap.walkable(p)]
    for i, tpl in enumerate(SQUAD_TEMPLATE):
        pos = spots[i % len(spots)]
        h = Entity(name=tpl["name"], pos=pos, hp=tpl["hp"], max_hp=tpl["hp"],
                   atk=tpl["atk"], kind="hero", role=tpl["role"])
        if h.role == "Healer":
            h.charges["heal"] = 5
        if h.role == "Tank":
            h.cooldowns["taunt"] = 0
        if h.role == "Scout":
            h.cooldowns["scout"] = 0
        heroes.append(h)

    free = [p for p in gmap.find_all(".")
            if manhattan(p, entry) > 8 and p not in {h.pos for h in heroes}]
    rng.shuffle(free)
    monsters: list[Entity] = []
    for i in range(min(n_monsters, len(free))):
        tpl = MONSTER_TEMPLATE[i % len(MONSTER_TEMPLATE)]
        mon = Entity(
            name=f"{tpl['prefix']}-{i + 1}", pos=free[i], hp=tpl["hp"],
            max_hp=tpl["hp"], atk=tpl["atk"], kind="monster",
            role=tpl["role"], speed=tpl["speed"])
        if mon.role == "Shaman":
            mon.charges["mend"] = 3
        monsters.append(mon)

    return GameState(gmap, heroes, monsters, seed=seed, max_turns=max_turns)


def explore_target(state: GameState, hero: Entity) -> Pos | None:
    blocked = frozenset(p for p in state.occupied if p != hero.pos)
    return nearest_frontier(state.gmap, state.known, hero.pos, blocked)
