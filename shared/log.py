"""CSV + stdout metric logger, run dirs, checkpoint save/load, manifest writer."""
import csv
import datetime
import hashlib
import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def new_run_dir(module: str, root: Path = None) -> Path:
    root = root or (REPO_ROOT / "runs")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{module}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = root / f"{module}-latest"
    latest.unlink(missing_ok=True)
    try:
        latest.symlink_to(run_dir.name)
    except OSError:
        pass
    return run_dir


class CSVLogger:
    """Append metric rows to <run_dir>/metrics.csv and echo to stdout every `echo_every`."""

    def __init__(self, run_dir: Path, echo_every: int = 1):
        self.path = Path(run_dir) / "metrics.csv"
        self.echo_every = echo_every
        self._fields = None
        self._n = 0

    def log(self, **metrics):
        metrics = {"step": metrics.pop("step", self._n), **metrics}
        new_file = self._fields is None
        if new_file:
            self._fields = list(metrics)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            if new_file:
                w.writeheader()
            w.writerow(metrics)
        self._n += 1
        if self._n % self.echo_every == 0:
            print("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))


def save_ckpt(path: Path, model: torch.nn.Module, optimizer=None, step: int = 0, extra: dict = None):
    payload = {"model": model.state_dict(), "step": step, "extra": extra or {}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    tmp = Path(path).with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.rename(path)


def load_ckpt(path: Path, model: torch.nn.Module, optimizer=None) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def write_manifest(run_dir: Path, manifest: dict, validate: bool = True):
    """Write manifest.json; validates against runs_manifest.schema.json when jsonschema is installed."""
    if validate:
        try:
            import jsonschema

            schema = json.loads((REPO_ROOT / "runs_manifest.schema.json").read_text())
            jsonschema.validate(manifest, schema)
        except ImportError:
            pass
    path = Path(run_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
