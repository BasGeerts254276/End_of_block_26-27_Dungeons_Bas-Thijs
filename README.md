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
python eval.py                                            # full benchmark
python tests/test_smoke.py
```

For LLM mode:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export WARBAND_MODEL=claude-haiku-4-5-20251001   # optional
```

Without a key everything still runs on the scripted policies. That is
deliberate: there is always a demo that works.

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
python eval.py                                   # 600 games, ~5 min
python eval.py --runs 20 --out small.csv         # quick 120-game check
python eval.py --conditions keeper_on --out on.csv   # one condition
python eval.py --from-csv on.csv off.csv --plot results.png   # merge chunks
```

Results over 600 games (3 maps x 100 seeds x 2 conditions), both sides
scripted:

| Condition | n | Squad win rate | Turns to win | Heroes lost | Chests |
|---|---|---|---|---|---|
| Keeper on | 300 | 57.0% [51–62] | 51.6 | 1.41 | 77% |
| Keeper off | 300 | 66.0% [60–71] | 51.3 | 1.40 | 82% |

Two-proportion z-test: z = -2.27, p = 0.024. The enemy commander costs the
squad nine percentage points of win rate, and that is significant.

Note the 95% intervals in the plot overlap slightly while the test says
significant. Overlapping error bars are not a significance test; the
two-proportion test is. Reading the bars by eye would have given the wrong
answer here.

### What the commander actually changes

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

`results.csv` holds one row per game with all of it, so you can slice by map
or seed yourself.

### Benchmarking LLM mode

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
