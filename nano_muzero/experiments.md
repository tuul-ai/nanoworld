# nano_muzero break-it protocol (M1.g, scene 1.9)

Each experiment states its expected qualitative outcome BEFORE you run it, and what result
would falsify the module's claims. Everything runs seeded on a Mac CPU; fill your numbers in
next to ours. Commands assume the repo root and a built baseline (`python -m
nano_muzero.baseline`).

## (a) K = 1 retrain: does unrolled training actually matter?

```
python -m nano_muzero.train --full --unroll 1 --tag k1 --seed 0
python -m nano_muzero.eval --ckpt data/ckpts/muzero_selfplay_k1.pt --gates full
python -m nano_muzero.eval --ckpt data/ckpts/muzero_selfplay.pt   --gates full   # K=5 reference
```

- **Expected:** the K=1 agent's arena results drop relative to K=5, and
  `python -m nano_muzero.eval_drift` (offline checkpoints) shows its value drift exploding
  past depth ~4 while K=5 stays flat to its trained depth.
- **Falsifier:** K=1 matching K=5 everywhere would mean composing g was never the hard part,
  and scene 1.5's "train through the composition" beat is theater.
- **Ours (offline drift, |v^k - z|, 50 positions):** K=1 hits 0.56-0.59 at depths 6-7 where
  K=5 stays 0.11-0.27 (see `data/eval/drift_curves.json`).

## (b) Noisy Tic-Tac-Toe: what a deterministic g does to a stochastic world

```
python -m nano_muzero.train --full --env noisy --tag noisy --seed 0
python -m nano_muzero.eval --ckpt data/ckpts/muzero_selfplay_noisy.pt --gates full
```

- **Expected:** with p = 0.1 random landings, the deterministic g learns a blur; value
  calibration (gate 3, and the eval table generally) visibly degrades vs the clean run.
  This is failure mode 2 of scene 1.9; Stochastic MuZero's afterstates are the fix.
- **Falsifier:** a noisy-env agent as well-calibrated as the clean one would mean
  determinism was never a real constraint at this scale.

## (c) Value-blind latent: kill the only teacher

```
python -m nano_muzero.train --offline --unroll 5 --ablate-value
python -m nano_muzero.probe
```

- **Expected:** with value AND reward losses off, the latent is shaped by policy alone;
  the probe's value row collapses to chance while occupancy decodability persists partially
  (policy still needs some board). Failure mode 4: no reward signal, no value-shaped latent.
- **Falsifier:** a value-blind latent whose value head still reads accurately would break
  scene 1.4's claim that the value information survives *by construction*.

## Results table (fill in per run; seeds printed by each command)

| experiment | metric | clean/K=5 reference | broken run |
|---|---|---|---|
| (a) K=1 | drift at depth 6 | 0.265 | 0.564 |
| (b) noisy | gate table | see eval output | see eval output |
| (c) value-blind | probe value sign acc | 0.991 | 0.568 (occupancy stays 0.927) |
