# Decisions ledger

Blocking inputs from PLAN.md, each resolved inside its plan step. Record the ruling + evidence
here (and in the deck colophon where relevant); nothing ships against an unresolved decision.

| id | question | resolved at step | ruling |
|---|---|---|---|
| D1 | value-head convention (derived from the shipped capstone code) | 18 | **RESOLVED 2026-07-02:** value in **[-1, 1] via tanh, targets z in {-1, 0, +1} from the mover's view, MSE loss.** The capstone ships as a spec, not code (`courses/alphazero/capstone/` holds only README.md + AGENT.md; path 3 "hand it to your agent" makes AGENT.md the source of truth), and AGENT.md pins the convention three times: value "in [-1, 1]" (line 9), value head "-> 1 -> tanh (in [-1, 1], from the to-play view)" (line 59), `loss = cross_entropy + mse(v, z)` with `z * player` labels (lines 96, 106). `nano_muzero/baseline.py` (this repo's concrete capstone re-run, M1.a) implements exactly that: `AlphaZeroNet.value_head` tanh, `play_game` z*player labels, `train` MSE. The AlphaZero deck's [0,1]+BCE is a deck-widget convention and ships in Module 1 prose as the labeled aside (module doc's notation table updated accordingly: deck scenes show tanh/MSE as the code convention). |
| D2 | watch-nano-muzero-learn: live in-browser training vs labeled recording | 26 | **RESOLVED 2026-07-02: hybrid.** The LIVE part is the offline M1.d phase: real JS backprop (hand-rolled for the fixed MLP stack, latent 8) training h/g/f on a shipped slice of the frozen replay, with loss + drift + arena-vs-random improving in seconds; genuinely learnable in-browser and mirrors the milestone the learner just ran. The full self-play flywheel ships as a LABELED RECORDING of the real Python run (metrics.csv + snapshot arena curves) because measured nano-scale self-play needs the mix-replay/coverage recipe and minutes of compute; the delusion finding (X-wins collapse) made a live from-scratch flywheel an honesty risk, not just a perf risk. The recording is captioned as a recording, per the module doc's fallback language. |
| D3 | Dreamer action space (recommendation: proven 7-D joint interface) | 34 | OPEN |
| D4 | T5 peg-insertion asset: build vs "why this is hard" essay tier | 37 | OPEN |
| D5 | HIW nano-slice download-size target + subtask-label whitelist | 48 | OPEN |
| D6 | reference-annotation budget; Gemini-Flash-assisted first pass acceptable? | 54 | OPEN |
| D7 | bridge-module placement (after M4 vs after M1) | 64 | OPEN |
| D8 | nano_genie codebook: copy 8 vs reduced-size {4,8,16} sweep | 59 | OPEN (default: no full-size sweep) |
| D9 | learner-aggregate annotation ratings backend write path | 56 | OPEN (v1 fallback: author-reference-only) |
| D10 | mobile tier for heavy labs | Phase 0 gate | **RESOLVED 2026-07-02 (Shreyas):** Tier 2 labs are desktop-only; on touch/small viewports each shows a poster video + "open on desktop", never a broken canvas. Module list + palette + satellite schema confirmed at the same gate. |

## Gate rulings

- **M1.e never-loses gate (2026-07-02, Shreyas):** the 18/200 losses-to-random tail is
  ACCEPTED as course content (option a). The other four gates pass (arena net +23 vs
  mirror +4; root value +0.087; oracle 20/20; falsifier +33 vs +17). The tail is
  characterized in nano_muzero/README.md + experiments.md and feeds the 1.8 lab's
  hallucination finder. Phase 1 code gate signed off; deck authoring unlocked (PR #1).
