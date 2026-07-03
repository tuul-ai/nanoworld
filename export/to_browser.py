"""Export trained checkpoints to the browser: weights JSON + golden vectors (step: export).

  python export/to_browser.py --module muzero [--ckpt data/ckpts/muzero_selfplay.pt]

Writes:
  export/out/muzero/weights_<name>.json    -- {config, params}, consumed by export/js/forward.js
                                              (and the deck's 1.8 lab "upload your weights")
  export/out/muzero/manifest.json          -- provenance: source ckpt sha256 + run config
  export/golden/muzero_weights.json        -- committed CI fixture (small net is fine)
  export/golden/muzero_golden.json         -- fixed input -> PyTorch outputs; node asserts
                                              the JS forward pass reproduces them exactly:
                                              the thing in the browser IS the thing you trained

Then prove it: node export/js/check_golden.mjs muzero
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from envs.tictactoe import TicTacToe  # noqa: E402
from nano_muzero.baseline import AlphaZeroNet  # noqa: E402
from nano_muzero.model import MuZeroNet  # noqa: E402

OUT_DIR = REPO_ROOT / "export" / "out"
GOLDEN_DIR = REPO_ROOT / "export" / "golden"

# a fixed midgame board (X about to fork) as the golden input, plus a fixed action
GOLDEN_BOARD = (1, 0, 0, 0, -1, 0, 0, 0, 1)
GOLDEN_ACTION = 2


def load_muzero(ckpt: Path) -> tuple:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = {"obs_dim": 18, "latent_dim": payload["extra"]["latent_dim"],
           "hidden": payload["extra"]["hidden"], "n_actions": 9}
    net = MuZeroNet(latent_dim=cfg["latent_dim"], hidden=cfg["hidden"])
    net.load_state_dict(payload["model"])
    net.eval()
    return net, cfg


def weights_payload(net: MuZeroNet, cfg: dict) -> dict:
    params = {k: {"shape": list(v.shape), "data": v.detach().float().flatten().tolist()}
              for k, v in net.state_dict().items()}
    return {"config": cfg, "params": params}


def golden_payload(net: MuZeroNet, weights_file: str) -> dict:
    """Chained golden: initial() on a fixed board, then recurrent() on its latent."""
    game = TicTacToe()
    obs = torch.from_numpy(game.encode(GOLDEN_BOARD)).unsqueeze(0)
    with torch.no_grad():
        s0, p0, v0 = net.initial(obs)
        s1, r1, p1, v1 = net.recurrent(s0, torch.tensor([GOLDEN_ACTION]))
    return {
        "weights_file": weights_file,
        "board": list(GOLDEN_BOARD),
        "obs": obs.flatten().tolist(),
        "action": GOLDEN_ACTION,
        "initial": {"s": s0[0].tolist(), "logits": p0[0].tolist(), "v": float(v0[0])},
        "recurrent": {"s": s1[0].tolist(), "r": float(r1[0]),
                      "logits": p1[0].tolist(), "v": float(v1[0])},
        "tolerance": 1e-5,
    }


def export_muzero(args):
    out = OUT_DIR / "muzero"
    out.mkdir(parents=True, exist_ok=True)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    ckpts = {"final": args.ckpt}
    snap_dir = args.ckpt.parent / "snapshots" if args.ckpt.name == "ckpt.pt" else None
    if snap_dir and snap_dir.is_dir():  # run-dir export: include the scrubber series
        for p in sorted(snap_dir.glob("iter_*.pt")):
            ckpts[p.stem] = p
    for name, path in ckpts.items():
        net, cfg = load_muzero(path)
        wpath = out / f"weights_{name}.json"
        wpath.write_text(json.dumps(weights_payload(net, cfg)))
        written.append(wpath)

    # the capstone net too: the 1.8 lab's TRUE-search side runs it in the browser
    cap_ckpt = REPO_ROOT / "data" / "ckpts" / "ttt_capstone.pt"
    if cap_ckpt.exists():
        payload = torch.load(cap_ckpt, map_location="cpu", weights_only=False)
        cap = AlphaZeroNet(hidden=payload["extra"]["hidden"])
        cap.load_state_dict(payload["model"])
        cap_cfg = {"obs_dim": 18, "hidden": payload["extra"]["hidden"], "n_actions": 9}
        (out / "weights_capstone.json").write_text(json.dumps(
            {"config": cap_cfg, "params": {k: {"shape": list(v.shape), "data": v.detach().float().flatten().tolist()}
                                            for k, v in cap.state_dict().items()}}))
        written.append(out / "weights_capstone.json")

    # CI fixture pair: weights + golden vectors from the FINAL net
    net, cfg = load_muzero(args.ckpt)
    (GOLDEN_DIR / "muzero_weights.json").write_text(json.dumps(weights_payload(net, cfg)))
    (GOLDEN_DIR / "muzero_golden.json").write_text(
        json.dumps(golden_payload(net, "muzero_weights.json")))
    written += [GOLDEN_DIR / "muzero_weights.json", GOLDEN_DIR / "muzero_golden.json"]

    (out / "manifest.json").write_text(json.dumps({
        "module": "muzero",
        "source_ckpt": str(args.ckpt),
        "source_sha256": hashlib.sha256(args.ckpt.read_bytes()).hexdigest(),
        "config": cfg,
        "files": [str(p.relative_to(REPO_ROOT)) for p in written],
    }, indent=2))
    for p in written + [out / "manifest.json"]:
        print(f"wrote {p.relative_to(REPO_ROOT)}")


def export_deck_bundle(deck_dir: Path, ckpt: Path):
    """Everything the deck's Module 1 labs consume, in one directory:
    - weights_final/iter_*/capstone.json (from export_muzero's out dir)
    - lab-engine.js: forward.js + search.js concatenated, import/export stripped,
      exposed as window.NanoMuZero (the deck is a single file; labs lazy-load this)
    - replay_slice.json: 60 games from the frozen replay (watch-it-learn + hgf-io truth)
    - drift_curves.json + probe_latents.json (subsampled) for the 1.4/1.5 widgets
    """
    import numpy as np

    from nano_muzero import baseline

    deck_dir.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "muzero"
    for f in out.glob("weights_*.json"):
        (deck_dir / f.name).write_text(f.read_text())

    js = []
    for name in ("forward.js", "search.js"):
        t = (REPO_ROOT / "export" / "js" / name).read_text()
        t = t.replace('import { AlphaZeroJS, MuZeroJS } from "./forward.js";', "")
        t = t.replace("export function", "function").replace("export const", "const")
        t = t.replace("export class", "class")
        js.append(t)
    js.append("window.NanoMuZero = { linear, ACTS, mlpForward, rescaleLatent, MuZeroJS, "
              "AlphaZeroJS, TTT, MuZeroCfg, capstoneEquivalent, runMCTS, MuZeroModelJS, "
              "OracleModelJS, runCapstoneMCTS };")
    (deck_dir / "lab-engine.js").write_text("\n".join(js))

    data = baseline.validate_replay(baseline.REPLAY_PATH)
    keep = data["game_id"][:] < 60
    slice_ = {k: data[k][:][keep].tolist() for k in ("board", "to_play", "action", "z", "game_id", "move_idx")}
    slice_["pi"] = np.round(data["pi"][:][keep], 4).tolist()
    (deck_dir / "replay_slice.json").write_text(json.dumps(slice_))

    drift = REPO_ROOT / "data" / "eval" / "drift_curves.json"
    if drift.exists():
        (deck_dir / "drift_curves.json").write_text(drift.read_text())
    probe = REPO_ROOT / "data" / "eval" / "probe_latents.npz"
    if probe.exists():
        d = np.load(probe)
        idx = np.arange(0, len(d["obs"]), max(1, len(d["obs"]) // 800))
        (deck_dir / "probe_latents.json").write_text(json.dumps({
            "labels": d["labels"][idx].tolist(),
            "z": d["z"][idx].tolist(),
            "latents_off": np.round(d["latents_off"][idx], 5).tolist(),
            "latents_on": np.round(d["latents_on"][idx], 5).tolist(),
        }))
    print(f"deck bundle -> {deck_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--module", choices=["muzero"], required=True)
    ap.add_argument("--ckpt", type=Path,
                    default=REPO_ROOT / "data" / "ckpts" / "muzero_selfplay.pt")
    ap.add_argument("--deck", type=Path, default=None,
                    help="also write the deck asset bundle to this directory")
    args = ap.parse_args(argv)
    export_muzero(args)
    if args.deck:
        export_deck_bundle(args.deck, args.ckpt)


if __name__ == "__main__":
    main()
