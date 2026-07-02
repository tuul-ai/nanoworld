# nanoworld

The nanoGPT of world models. One question runs through the whole repo: **where does the simulator come from?**

Four nano models answer it in sequence, each small enough to read in a sitting:

| module | paper anchor | the answer it gives |
|---|---|---|
| `nano_muzero/` | MuZero, arXiv:1911.08265 | learn the simulator's *transitions* and plan inside them (tic-tac-toe) |
| `nano_dreamer/` | World Models arXiv:1803.10122; DreamerV3 arXiv:2301.04104 | learn a *latent* simulator from pixels and train the policy inside the dream (YAM arm sim) |
| `nano_jepa/` | I-JEPA arXiv:2301.08243 | predict in representation space, skip the pixels (real-home robot video, HIW-500) |
| `nano_genie/` | Genie, arXiv:2402.15391 | learn the simulator *and its action space* from raw video |

This repo is the code companion of the tuul.dev **worldmodels** course, the way
`courses/alphazero/capstone/` is the companion of the AlphaZero course. Every number the course
quotes from here traces to a run manifest (`runs_manifest.schema.json`).

## Layout rules (binding)

- **No cross-module imports.** `nano_dreamer` never imports from `nano_muzero`. Only `shared/`,
  `envs/`, `data/` are importable by modules. Duplication between modules is acceptable and
  pedagogically intended (nanoGPT rule).
- **`shared/` grows only by extraction.** Something enters `shared/` only after two modules have
  written it independently. Target under ~600 lines total.
- **One idea per file**, ~300-600 lines for a `model.py`, no framework, no config zoo.
  Hyperparameters are a plain dataclass at the top of `train.py` with a `tiny` (CPU/MPS) and
  `real` (Modal) preset.
- **READMEs are course material.** Each module README opens with what its training data actually
  looks like (real sample frames and episodes), then follows the capstone template: what you're
  building, per-file spec, checks after each piece, milestones, known-bug callouts, and a
  "what would falsify this?" section.

## The embodiment jump (a stated teaching point)

The course moves game -> YAM sim arm -> real G1 homes, but the modeling machinery (h/g/f, RSSM,
JEPA predictor, latent actions) never touches embodiment-specific code; only `envs/` and `data/`
do. The simulated robot is the **YAM arm** (`envs/yam_pickplace.py`, ported from the proven
caferacer sim). The Unitree G1 appears only as **real data** via `data/hiw500.py`; there is no G1
sim here. One episode schema (obs, action, reward, done) is shared by the tic-tac-toe self-play
recorder, the YAM recorder, and the HIW shards, so `shared/buffer.py` never cares where data
came from.

## Spend discipline

Every paid launch goes through `scripts/modal_guard.py`, which reads the `BUDGET.md` ledger,
projects the run's cost, and refuses to launch past the phase cap. Every run writes
`runs/<id>/manifest.json` conforming to `runs_manifest.schema.json`. Always
`modal run --detach`; hard timeout on every function; A10G default; a `--steps 500` paid dress
rehearsal before every long run.

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -e ".[dev]"
venv/bin/pytest -q
```

Extras: `[sim]` MuJoCo 3.10.0 + imageio, `[data]` HuggingFace hub + parquet, `[export]`
onnx + onnxruntime, `[modal]` Modal.
