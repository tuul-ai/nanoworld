# Modal budget ledger

Hard ceiling **$300** including contingency; planned spend **$240**. Update the ledger after
EVERY paid run with the actual $ from the Modal dashboard. `scripts/modal_guard.py` parses the
two tables below and refuses launches that would push a phase past its cap. If cumulative
actuals hit 80% of a phase cap, stop and re-scope before launching the next run.

## Phase caps

| phase | scope | cap_usd |
|---|---|---|
| 0 | scaffold + spikes | 0 |
| 1 | MuZero (all Mac MPS/CPU) | 0 |
| 2 | Dreamer: N4 world-model run, KL-balance triplet, N8 multi-task, one redo reserve | 80 |
| 3 | JEPA: HIW prep, nano_jepa run, nano_jepa_ac run | 70 |
| 4 | Genie: tokenizer + LAM + dynamics stagewise, reduced codebook sweep if D8 funds it | 90 |
| 5 | bridge/satellites (GRPO stays paper-and-widget in v1) | 0 |

Contingency $60 on top of the $240 planned total; the guard enforces per-phase caps, the $300
ceiling is the sum check.

## Prices (re-verified live 2026-07-02 against modal.com/pricing)

| resource | per second | per hour |
|---|---|---|
| Nvidia A10 | $0.000306 | $1.10 |
| Nvidia A100 40GB | $0.000583 | $2.10 |
| Nvidia A100 80GB | $0.000694 | $2.50 |
| Nvidia H100 | $0.001097 | $3.95 |
| Nvidia T4 | $0.000164 | $0.59 |
| Nvidia L4 | $0.000222 | $0.80 |
| CPU physical core | $0.0000131 | $0.047 |
| Memory per GiB | $0.00000222 | $0.008 |
| Volume storage | | $0.09/GiB-month, 1 TiB/mo free |

These match the plan's pre-run arithmetic (PLAN.md budget section); no cap rework needed.
Note: Modal's table lists "A10" (the guard accepts `a10g` as an alias). GPU price excludes the
CPU/memory attached to the container; the guard adds a 15% overhead factor for that.

## Pre-run arithmetic (from PLAN.md, projections not actuals)

- HIW prep: ~15 core-hours x3 overhead, ~$2-10 (confirm from measured sample throughput at step 46)
- nano_jepa: ~2.1e17 FLOPs, ~3 A10-hours, ~$3-30
- nano_genie dynamics: ~1.8e18 FLOPs, ~25 A10-hours (~$28) or ~10 A100-hours (~$21); full-size codebook sweep triples this, so D8 defaults to reduced or none
- Storage: ~40 GB shards inside the 1 TiB free tier, ~$0

## Ledger (append one row per paid run)

| date | phase | run_id | gpu | hours | actual_usd | note |
|---|---|---|---|---|---|---|
