"""One Modal app for all of nanoworld; per-module entrypoints land with their phases.

Rules (see README + BUDGET.md):
- Entrypoints are THIN: all logic lives in each module's train.py, which runs identically
  locally (`python -m nano_dreamer.train`) and here. This file only wires volumes and env vars.
- Every launch goes through scripts/modal_guard.py, always `modal run --detach`.
- Hard timeout on every function; A10G default; print GPU type + step budget at run start.
- Checkpoints resumable; ckpt_vol.commit() after saves.

Phase 2 adds train_dreamer, Phase 3 prep_hiw500 + train_jepa, Phase 4 train_genie.
"""
import modal

app = modal.App("nanoworld")

image = modal.Image.debian_slim().pip_install("torch", "numpy").add_local_python_source(
    "shared", "envs", "data", "nano_muzero", "nano_dreamer", "nano_jepa", "nano_genie"
)

data_vol = modal.Volume.from_name("nanoworld-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("nanoworld-ckpts", create_if_missing=True)


@app.function(image=image, timeout=300)
def smoke():
    """Wiring check only: proves the image builds and volumes mount. Free-tier trivial."""
    import torch

    print("nanoworld modal app alive; torch", torch.__version__)
