# Warband — Agentic Dungeon

Two agent networks fighting over a dungeon.

A squad of four adventurers, each with its own system prompt and its own tool
definitions, coordinated by a planner. Against them, a Dungeon Keeper
commanding three kinds of monsters. Both sides only know what they can see.
Underneath every agent sits a deterministic scripted policy that is both the
safety net and the measurement baseline.

## Quick start

```bash
pip install -r requirements.txt

python main.py --map maps/warren.txt --mode scripted      # windowed
python main.py --map maps/crypt.txt --mode llm --fps 4    # with the LLM
python main.py --mode scripted --headless --seed 7        # no window
python main.py --mode scripted --no-keeper                # enemy commander off
python main.py --map maps/warren.txt --seed 1 --paused    # step through a game
python main.py --mode scripted --no-smoke                 # Scout without smoke
python eval.py --conditions all --runs 100                # full benchmark (1200 games)
python tests/test_smoke.py
```

For LLM mode:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export WARBAND_MODEL=claude-haiku-4-5-20251001   # optional
```

Without a key everything still runs on the scripted policies. That is
deliberate: there is always a demo that works.

In the window: **space** pauses and resumes, **n** (or the right arrow)
plays one turn while paused, **Esc** quits. `--fps 0.5` plays one turn every
two seconds.

## Layout

| File | Responsibility |
|---|---|
| `world.py` | Grid, line of sight, A\*, frontier detection. No LLM, no state. |
| `engine.py` | Entities, game state, **action validation**, perception for both factions, prompt serialisation. |
| `agents.py` | Four squad personas: system prompt, per-role tool schemas, scripted policy. |
| `planner.py` | Squad commander. One LLM call every 3 turns handing out goals. |
| `enemy.py` | Dungeon Keeper plus three monster roles, and the ablation switch. |
| `llm.py` | Anthropic API wrapper. Never raises into the rest of the program. |
| `sim.py` | The turn loop tying it all together. |
| `render.py` | PyGame view with fog of war. Entirely optional. |
| `eval.py` | Batch runs over seeds, compares modes, writes CSV. |
| `tools/make_maps.py` | Generates and validates the maps in `maps/`. |
| `tests/test_smoke.py` | Tests, including the anti-cheat check. |

## The two agent networks

**The squad.** Tank, DPS, Healer and Scout. The Tank can taunt but not heal;
the Healer can heal but not taunt; only the Scout has the wide-area reveal.
The planner assigns a goal (`explore`, `engage`, `loot`, `regroup`,
`retreat`) every three turns and each persona works out how to pursue it.
The Scout also carries one smoke bomb, and the DPS can set that smoke on
fire: a two-agent combo (see [Smoke and ignite](#smoke-and-ignite)).

One honest caveat: the goals reach the personas through the LLM prompt
(`describe_for`). The scripted fallback policies do not read them, so in
scripted mode the planner hands out goals that nobody acts on. That is why
the commander ablation below is run on the enemy side, where the Keeper's
orders drive the monsters in both modes. The assignment only requires a
commander for the enemies.

**The enemy.** Brutes are slow and tough. Stalkers are fast and fragile and
deliberately target the lowest-HP hero rather than the nearest one. Shamans
mend wounded monsters and keep out of the front line. The Keeper issues
orders (`swarm`, `ambush`, `hunt`, `guard`, `flee`) on the same cadence.

## Nobody reads the whole map

Both sides have their own perception and their own memory. The squad has
`state.known` and `state.visible`; the enemy has `state.mon_known`,
`state.mon_visible` and `state.hero_last_seen`. Monster vision is 5 tiles,
squad vision is 6.

A\* refuses to route through tiles the acting faction has not discovered, so
agents cannot plan around walls they have never seen. When a hero slips out
of sight, the monsters keep only the memory of where it last was and hunt
that spot.

`test_enemy_never_reads_the_squad_map` walks the AST of `enemy.py` and fails
if it finds any access to `state.known` or `state.visible`. That is the
anti-cheat rule enforced in code rather than promised in prose.

## Smoke and ignite

Added in v2, because the Scout was the squad's second most frequent casualty
(44% of games with the Keeper on): the Stalkers deliberately hunt the
lowest-HP hero, and that is usually Pip.

**Smoke (Scout, one per game).** A cloud of up to 5x5 tiles (walls cut it
off). It stays at full size for three turns, then collapses to 3x3 for one
turn, then it is gone.

- Monsters cannot see into or through the cloud, and a monster standing in
  it is blind. They cannot target a hero inside, so they fall back on
  where they last saw him (`hero_last_seen`), and that hero is already
  gone.
- The squad can see through its own smoke. It knows where it threw it,
  and the DPS needs to see who is caught inside.
- Thrown on himself, the Scout also slips up to two tiles away from the
  nearest enemy, staying inside the cloud.
- It can also be thrown over a wounded ally within four tiles.
- "Ninja" bonus: any hero that starts its move inside smoke moves one
  extra tile, so whoever is hiding can actually get away.

**Ignite (DPS).** Sets a cloud within five tiles on fire: every monster
inside takes 6-10 damage, and the cloud burns away. That is a real
trade-off, because whoever was hiding in it is exposed again. The scripted
rule ignites when it catches two or more monsters, when the cloud is
thinning anyway, or when no ally is inside.

Smoke is implemented as an extra sight blocker (`fov(..., extra_blockers)`
in `world.py`) and the enemy's vision goes through `GameState.monster_fov()`.
Nothing about the enemy's knowledge model had to change: it still only
knows what it sees, the smoke just changes what it can see. All the numbers
are constants at the top of `engine.py` (`SMOKE_*`, `IGNITE_*`), and
`--no-smoke` turns the whole thing off for the ablation.

## Design choices that matter

**Validation over trust.** Every action goes through `GameState.apply()`,
which returns either a success or a reason for refusal. The model may ask to
attack an enemy three rooms away; the engine refuses with a readable reason
and the scripted policy takes over. `apply()` even catches exceptions, so a
malformed JSON argument cannot crash the simulation.

**Role differences in the API contract, not only the prompt.** The Healer is
simply never offered `taunt`. If the model asks for it anyway, that counts as
an invalid action and shows up in the metrics. That way you measure whether
your prompts work instead of hoping they do.

**Commanders do not run every turn.** One call every three turns on each
side. That saves roughly three quarters of the coordination cost and stops
both factions from changing strategy every single turn.

## Benchmark

`eval.py` runs the same fixed seeds under every condition, so a difference
between conditions cannot come from the dungeon layout or from where the
monsters spawned.

```bash
python eval.py --conditions all --runs 100       # 1200 games, all four conditions
python eval.py --runs 100                        # 600 games, Keeper on/off only
python eval.py --runs 20 --out small.csv         # quick 120-game check
python eval.py --conditions keeper_on --out on.csv   # one condition
python eval.py --from-csv on.csv off.csv --plot results.png   # merge chunks
```

There are four conditions: the enemy commander (Keeper) on or off, crossed
with the Scout's smoke on or off. `results.csv` and `results.png` hold the
current run.

### Current results (v2: smoke added, keeper_off vision fixed)

1200 games (3 maps x 100 seeds x 4 conditions), both sides scripted:

| Condition | n | Squad win rate | Turns to win | Heroes lost | Chests | Scout death rate |
|---|---|---|---|---|---|---|
| Keeper on, smoke | 300 | 63.3% [58–69] | 53.2 | 1.30 | 82% | 31% |
| Keeper off, smoke | 300 | 70.0% [65–75] | 54.1 | 1.21 | 86% | 28% |
| Keeper on, no smoke | 300 | 57.0% [51–62] | 51.6 | 1.41 | 77% | 44% |
| Keeper off, no smoke | 300 | 58.3% [53–64] | 51.7 | 1.28 | 80% | 42% |

Two-proportion z-tests:

| Comparison | Win rate | Scout death rate |
|---|---|---|
| Keeper on vs off, with smoke | −6.7 pts, p = 0.083 | +3.0 pts, p = 0.42 |
| Keeper on vs off, no smoke | −1.3 pts, p = 0.74 | +2.0 pts, p = 0.62 |
| Smoke vs none, Keeper on | **+6.3 pts**, p = 0.11 | **−12.3 pts, p = 0.002** |
| Smoke vs none, Keeper off | **+11.7 pts, p = 0.003** | **−13.3 pts, p = 0.0006** |

**Sanity check.** "Keeper on, no smoke" reproduces the v1 "Keeper on"
result exactly (57.0%, 51.6 turns, 1.41 heroes lost, 77% chests). That code
path was not touched, and the fixed seeds give identical games. So every
difference below comes from the smoke or from the vision fix, not from
noise.

### What smoke changes

- **It does what it was built for.** The Scout's death rate drops by about
  13 points in both Keeper conditions, which is significant in both.
- **It wins more games, but mostly when the monsters fight alone:**
  +11.7 points without the Keeper (significant), +6.3 with it (not
  significant). The most likely reason is the Keeper's shared memory. Alone,
  a monster that loses sight of a hero has nothing left to chase. Under the
  Keeper, `hero_last_seen` is shared and `hunt` and `swarm` send the whole
  pack to the last known spot, which is exactly where the cloud is. The
  commander partly counters the smoke.
- **The combos are rare.** Smoke is used in 61% of games, but almost always
  by the Scout on himself (0.59 per game; on an ally only 0.01-0.03).
  Ignite happens in 8% of games. The scripted Scout spends his only charge
  on himself before a teammate qualifies (below 40% HP, in melee, within
  four tiles). A second charge (`SMOKE_CHARGES = 2`) would be the obvious
  next experiment.
- The DPS is still the most frequent casualty in every condition (about
  half of all games). Smoke protects the Scout, not the DPS on the front
  line.

### Why the commander no longer makes a measurable difference

Without smoke, Keeper on and Keeper off are now within 1.3 points of each
other (p = 0.74). In v1 the same comparison was 9 points and significant.
Two things most likely explain this.

1. **The v1 ablation leaked.** In `disband()` (Keeper off) each monster
   detected heroes within five tiles by distance alone, through walls, and
   still picked targets from the faction's shared sight. That broke the
   "agents only know what they can see" rule in the very condition meant
   to show what monsters do without coordination. v2 gives every monster
   only its own field of view (`monster_fov`, walls and smoke included)
   while it acts. The squad's win rate without the Keeper then drops from
   66.0% to 58.3%, the same level as with the Keeper. Our reading: in v1 the
   lone monsters homed in on the squad through walls and trickled in one at
   a time (the v1 notes below describe exactly that), which made them easy
   kills. With honest vision they no longer do.
2. **"Keeper off" still has some coordination left.** When a lone monster
   sees no hero, `swarm` sends it to the centre of mass of all living
   monsters (`_rally_point`). So even without a commander the monsters
   regroup into a pack. On top of that, the Keeper's rule-based orders are
   80% `guard` and `swarm`, which is close to what monsters do on their own.
   The commander adds little that the default behaviour does not already
   do.

The Keeper still has a measurable side effect in the data: without smoke
the first hero dies earlier with the Keeper on (turn 18.4 versus 20.4), and
wipes are more common (12% versus 8%). It makes fights harder, just not
enough to move the win rate.

A cleaner ablation would make `_rally_point` only use monsters that the
acting monster can see. That is listed under possible extensions.

### Per map

| Condition | crypt | halls | warren |
|---|---|---|---|
| Keeper on, smoke | 69% | 60% | 61% |
| Keeper off, smoke | 82% | 71% | 57% |
| Keeper on, no smoke | 59% | 50% | 62% |
| Keeper off, no smoke | 60% | 62% | 53% |

Each cell is 100 games, so a single cell has an interval of roughly ±10
points: read this table for direction, not for conclusions. Smoke helps
most on crypt and barely on warren.

### First run (v1), kept for the record

These are the numbers from before smoke and before the `disband()` vision
fix. The Keeper-off condition in this run could sense heroes through walls,
so the ablation result below is not valid anymore. It is kept because it
shows how the conclusion changed, and the raw data is in `results_v1.csv`
and `results_v1.png`.

Results over 600 games (3 maps x 100 seeds x 2 conditions), both sides
scripted:

| Condition | n | Squad win rate | Turns to win | Heroes lost | Chests |
|---|---|---|---|---|---|
| Keeper on | 300 | 57.0% [51–62] | 51.6 | 1.41 | 77% |
| Keeper off | 300 | 66.0% [60–71] | 51.3 | 1.40 | 82% |

Two-proportion z-test: z = -2.27, p = 0.024. The enemy commander costs the
squad nine percentage points of win rate, and that is significant.

Note the 95% intervals in `results_v1.png` overlap slightly while the test says
significant. Overlapping error bars are not a significance test; the
two-proportion test is. Reading the bars by eye would have given the wrong
answer here.

#### What the commander appeared to change in v1

Coordination does not kill more heroes. Both conditions lose about 1.4 of
four per game. What changes is *how* the squad fails:

- Games lost to the turn limit rise from 24% to 31%. Guarding chests and
  ambushing works: the squad spends its turns fighting instead of looting,
  and chest completion drops from 82% to 77%.
- The first casualty comes later with the Keeper on (turn 18.4 versus 16.1).
  Uncoordinated monsters trickle in one at a time and die early; coordinated
  ones wait and strike together.
- The Tank's death rate falls from 34% to 29% while the Scout's rises from
  39% to 44%. That is the Stalker order working as intended: with shared
  intelligence the fast monsters go around the Tank and hunt the back line.

So the commander does not win the fight. It wins the clock, and it picks
better targets.

### Metrics reported

Per condition: win rate with a Wilson 95% interval, mean turns to win with
spread, heroes lost per game, chest completion, monsters killed, the split
between wipes and timeouts, the turn of first death, deaths per role, and
the invalid-action rate. That last one is the prompt-quality signal in LLM
mode; in scripted mode it should be a flat 0%.

Since v2 also: Scout death rate, smokes used (and how many on an ally),
ignites, burn damage and burn kills, with z-tests on both win rate and
Scout death rate for every pair of conditions.

`results.csv` holds one row per game with all of it, so you can slice by map
or seed yourself.

### Benchmarking LLM mode

We had no API key, so this repo contains no LLM-mode results; LLM mode is
covered by tests with a mocked model only.

`--mode llm` works but costs roughly 200 API calls per game. Six hundred
games is not realistic. Run a small sample on one map, report it next to the
scripted baseline, and say in the write-up that the sample is small.

## Deliberately out of scope

No procedural dungeon generator, no inventory, no spells, no sound. Those
cost time and earn no marks on an assignment about agent architecture.

## Possible extensions

- Short-term memory per agent: the last three actions in the prompt so agents
  stop repeating themselves.
- A real blackboard: agents leaving messages for their faction.
- Prompt ablations: run the same seeds with and without the role-specific
  instructions and measure the difference.
- Caching on identical game states to avoid repeated calls.
- A cleaner Keeper-off condition: `_rally_point` only over monsters the
  acting monster can see, so lone monsters cannot regroup by "knowing"
  where the pack is.
- Let the scripted personas read the planner's goals, so the squad
  commander also has an effect in scripted mode and can be ablated.
- A second smoke charge, to see whether the ally-smoke and ignite combos
  happen often enough to matter.
