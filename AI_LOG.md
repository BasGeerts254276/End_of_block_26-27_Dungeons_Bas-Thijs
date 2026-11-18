# AI_LOG.md

> Template note: this log reflects one author's sessions. The second team
> member should add their own sessions and overrides in the same format
> before handing in, and any claim below that does not match what you
> actually did should be edited rather than left standing.

## Tools used

Claude (Opus) through the claude.ai chat interface, used heavily for
implementation. No local models. No AI tooling inside the editor; code was
produced in chat, read, run locally on Windows (Python 3.13, pygame 2.6.1)
and then edited by hand where needed.

The honest summary: most of the Python in this repo was drafted by Claude.
Our contribution was scoping, judgement about what to keep, catching the
things it got wrong, and the decisions listed under "What we overrode"
below.

## What we asked, session by session

**1. Feasibility.** Before writing anything we asked how hard the Y2 Warband
track actually is and what a realistic hour estimate was for two people with
AI help. The first estimate came back at 100-140 team hours, which did not
fit a one-week challenge. We pushed back with the actual constraint and got
a revised plan that cut scope: no procedural generator, no real eval harness
at first, hand-made maps. That exchange set the architecture more than any
later one.

**2. First implementation.** We asked for the project rather than a single
file, because the assignment is graded on agent architecture. We specified
the split ourselves: perception and pathfinding separate from game state,
personas separate from the turn loop, rendering optional.

**3. Enemy commander.** After the deliverables post was published we asked
for the enemy side to be rebuilt: a commander, monster roles with different
abilities, and proper fog of war for the enemy faction.

**4. Translation to English.** The whole repo was first written in Dutch
(comments, prompts, log lines). We had it converted once we decided the
submission had to be in English.

**5. Benchmark harness.** We asked for the 100+ game benchmark with the
commander ablation, CSV output, and a plot.

**6. Smoke and ignite (v2).** Watching games, we noticed the Scout dying
a lot (44% of games with the Keeper on). We came up with two options
ourselves and asked Claude for pros and cons of each: a one-shot trap with
a "cry" that alerts the Keeper, or a one-use smoke bomb. We picked smoke,
and set the design: three turns instead of one, smoke-and-step-away in one
action, the option to throw it over a teammate, and a DPS ability to
ignite the smoke for damage to every monster inside. Claude implemented it
and extended `eval.py` to four conditions (Keeper on/off x smoke on/off).

**7. Making it watchable, and the ninja version.** In the window the smoke
was nearly invisible, so we asked for a stronger colour and a slower way
to watch. We got pause and single-step keys and fractional `--fps`, and
chose a toxic neon green ourselves. After watching it again we asked for a
bigger cloud that collapses fast after three turns and lets the squad get
away quickly. That became the 5x5 cloud, the one-turn collapse to 3x3,
the two-tile slip for the Scout and +1 move for anyone starting in smoke.

**8. Reading the v2 benchmark.** We ran 1200 games and asked for help
reading the report and writing it up. The write-up in the README is based
on that report plus a per-map breakdown and the Keeper's order mix.

## What we kept

- The layered structure: `world.py` (pure geometry) -> `engine.py` (state and
  validation) -> `agents.py` / `enemy.py` (policies and prompts) ->
  `sim.py` (loop). This survived every rewrite.
- Validation-before-trust in `GameState.apply()`. Every action, LLM or
  scripted, returns either success or a reason for refusal.
- The scripted fallback layer doing double duty as safety net and baseline.
  This was suggested early and turned out to be the single most useful idea
  in the project: it meant we always had a demo that ran, even with no API
  key, and it gave the benchmark something to measure against.
- Role differences encoded in the tool schemas rather than only in the
  prompts, so a Healer attempting to taunt is a measurable event.
- Commanders running every three turns instead of every turn.
- Smoke as an extra sight blocker inside the existing field-of-view code,
  rather than a special case in the monster AI. The enemy's knowledge model
  did not have to change; only what it can see did.
- The squad seeing through its own smoke. Without that the DPS could not
  tell who is caught in the cloud, and ignite would be guesswork.
- `--no-smoke` as a switch, so smoke is its own ablation on the same seeds
  instead of a before/after comparison of two different code versions.

## What we threw away, and why

**Hand-drawn ASCII maps.** The first three maps were written by hand. Two of
them had a sealed starting room: the squad could not leave and every run
ended at the turn cap with zero chests. We replaced them with
`tools/make_maps.py`, which carves rooms and corridors and then flood-fills
from the entrance to assert every chest is reachable. The generator is a
build tool, not a runtime feature; the `.txt` files are still the artefact.

**The original monster AI.** The first version pathfound to hero positions
using the complete map and targeted anyone within radius regardless of line
of sight. It worked, and it broke the "agents only know what they can see"
rule outright. Thrown away and replaced with a separate enemy perception
layer (`mon_known`, `mon_visible`, `hero_last_seen`). We added
`test_enemy_never_reads_the_squad_map`, which parses the AST of `enemy.py`
and fails if it finds any access to the squad's perception. We would rather
have the rule enforced by a test than promised in a README.

**Scout-only looting.** Looting was initially a Scout-exclusive tool, which
read nicely as role differentiation. In practice a DPS would wander onto a
chest tile, be unable to loot it, and block the Scout standing next to it.
Every remaining run deadlocked there. We gave `loot` to all four roles and
added position-swapping between allies in `_step_towards`. Role
differentiation now lives in taunt, heal, focus_fire and scout, which do not
have this failure mode.

**The first monster stat block.** When the Keeper was added, the existing
monster stats plus coordination wiped the squad in about twenty turns. Not a
bug, but an unusable benchmark: a floor-level win rate measures nothing. We
tuned HP and damage down until win rates landed in the 55-70% band, which is
where an ablation can actually show a difference. The tuning numbers were
chosen by running sweeps, not by asking for them.

**Overlapping confidence intervals as a significance test.** The first
version of the benchmark report compared the two conditions by checking
whether their 95% intervals overlapped, and concluded the ablation was
"suggestive, not proven". That is wrong: overlapping intervals do not imply
non-significance. We replaced it with a two-proportion z-test, which reports
p = 0.024 on the same data. The error bars in `results.png` still overlap
slightly, so this is noted explicitly in the README. We kept the plot rather
than hiding the awkward visual.

**The trap idea.** Claude's analysis convinced us it had too many moving
parts (trap object, trigger, cry, Keeper reaction, speed boost, squad
convergence) for one week, it would not directly protect the Scout, and
one trap per game would rarely trigger and be hard to measure. Dropped in
favour of smoke.

**One-turn smoke.** Our first idea. Heroes act before monsters, so a
one-turn cloud only blinds a single monster turn, and fast Stalkers would
see the Scout again immediately. Made it three turns.

**The grey and then light-blue cloud.** The first rendering was light grey
on a grey floor and almost invisible; light blue was better but did not
feel right. Toxic green it is.

**The v1 commander conclusion.** The v1 README said the Keeper costs the
squad nine points of win rate (p = 0.024). While adding smoke, Claude
found that `disband()`, the Keeper-off condition, let each monster detect
heroes within five tiles by distance alone, through walls, and still use
the faction's shared sight for targeting. That is cheating in exactly the
condition that is supposed to show monsters acting alone, and our AST
anti-cheat test did not catch it, because it only checks for
`state.known` and `state.visible`. With the leak fixed, the Keeper effect
without smoke shrinks to 1.3 points (p = 0.74). We threw the old
conclusion away, but kept the v1 numbers in the README, clearly labelled,
because how the conclusion changed is part of the result.

**Pressing n to stop a running game.** Claude offered to make `n` also
pause the game while it is running. We kept it simple: space pauses, `n`
steps while paused.

**An uninformative plot panel.** The middle panel originally showed heroes
lost per game, which is 1.41 versus 1.40 across conditions and says nothing.
Replaced with the outcome breakdown (win / wiped / out of turns), which is
where the commander's effect actually shows.

## What we overrode

- We rejected the initial scope and hour estimate as incompatible with a
  one-week challenge, and set the cut list ourselves.
- We insisted on the project being delivered a step at a time rather than in
  one dump, so each piece could be run and understood before the next was
  written.
- We decided the submission language, the deliverable format, and when to
  stop adding features.

## How we checked the output

We did not take working-looking code on trust. Specifically:

- `tests/test_smoke.py` has 18 tests that run without an API key. The LLM
  path is tested with a mocked model, so prompt plumbing, role restriction
  and fallback behaviour are all verified for free.
- The simulation was run on all three maps across many seeds after every
  structural change. Both the sealed-map bug and the loot deadlock were
  found this way, not by reading the code.
- `python tools/make_maps.py --check` validates map connectivity.
- The benchmark was run twice at different sample sizes (240 games, then
  600) to confirm the ablation result was not an artefact of n.
- After adding smoke, all 18 existing tests still passed, and we watched
  several games step by step with `--paused` to see the cloud appear,
  collapse and disappear.
- Regression check in the v2 benchmark: "Keeper on, no smoke" reproduces
  the v1 "Keeper on" numbers exactly (57.0%, 51.6 turns, 1.41 heroes lost),
  because that code path was not changed and the seeds are fixed. Every v2
  difference therefore comes from the smoke or from the vision fix.

## Known limitations

The benchmark numbers in the README are from scripted mode on both sides.
LLM mode works and is tested with a mocked model, but we had no API key
available, so this repo contains no LLM-mode results at all. Even with a
key, 1200 games at roughly 200 API calls each would not have been
affordable.

The v2 smoke results are also scripted on both sides. Two of the new
mechanics are rare in practice: smoke on an ally happens in 1-3% of games
and ignite in 8%, because the scripted Scout usually spends his single
charge on himself first. Those numbers are reported as they are, not tuned
until they looked good.

The squad planner only reaches the personas through the LLM prompt. In
scripted mode the fallback policies ignore its goals, so all scripted
benchmark numbers are effectively "squad planner off". The README says so.
