#!/usr/bin/env python3
"""Spend guard: every paid Modal launch goes through here, never around it.

Reads the BUDGET.md ledger (actual spend per phase) plus a projected cost for the run you are
about to launch, and refuses to launch if projected + actual would exceed the phase cap. At 80%
of a phase cap it warns loudly (the plan says: stop and re-scope). Writes a manifest stub
conforming to runs_manifest.schema.json so no run starts without one.

Usage:
  # check only (prints projected cost, PASS/FAIL, exit code 0/1):
  python scripts/modal_guard.py --dry-run --gpu a10g --hours 1 --phase 2

  # check, then launch detached and write the manifest stub:
  python scripts/modal_guard.py --launch "modal_app.py::train_dreamer --steps 500" \
      --gpu a10g --hours 0.5 --phase 2 --module nano_dreamer --seed 0
"""
import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUDGET = ROOT / "BUDGET.md"
HARD_CEILING_USD = 300.0

# $/hour, re-verified 2026-07-02 against modal.com/pricing (see BUDGET.md).
GPU_HOURLY = {
    "a10g": 1.10,
    "a10": 1.10,
    "a100-40gb": 2.10,
    "a100": 2.10,
    "a100-80gb": 2.50,
    "h100": 3.95,
    "t4": 0.59,
    "l4": 0.80,
    "cpu": 0.047,  # per physical core
}
# Container CPU/memory attached alongside the GPU, dress-rehearsal slop, etc.
OVERHEAD = 1.15


def parse_budget():
    """Return (caps: {phase: cap_usd}, spent: {phase: actual_usd})."""
    text = BUDGET.read_text()
    caps, spent = {}, {}
    section = None
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip().lower()
            continue
        m = re.match(r"^\|\s*(\d+)\s*\|(.+)\|\s*$", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if section and section.startswith("phase caps") and len(cells) >= 3:
            try:
                caps[int(cells[0])] = float(cells[2])
            except ValueError:
                pass
        elif section and section.startswith("ledger") and len(cells) >= 6:
            # | date | phase | run_id | gpu | hours | actual_usd | note |
            try:
                phase = int(cells[1])
                usd = float(cells[5])
            except ValueError:
                continue
            spent[phase] = spent.get(phase, 0.0) + usd
    if not caps:
        sys.exit("modal_guard: could not parse phase caps out of BUDGET.md")
    return caps, spent


def check(gpu, hours, phase, cores=1):
    caps, spent = parse_budget()
    if phase not in caps:
        sys.exit(f"modal_guard: unknown phase {phase} (caps exist for {sorted(caps)})")
    rate = GPU_HOURLY.get(gpu.lower())
    if rate is None:
        sys.exit(f"modal_guard: unknown GPU '{gpu}' (known: {', '.join(sorted(GPU_HOURLY))})")
    if gpu.lower() == "cpu":
        rate *= cores
    projected = round(rate * hours * OVERHEAD, 2)
    actual = spent.get(phase, 0.0)
    total_spent = sum(spent.values())
    cap = caps[phase]

    print(f"phase {phase} cap:        ${cap:.2f}")
    print(f"phase {phase} actuals:    ${actual:.2f}")
    print(f"projected this run: ${projected:.2f}  ({gpu} x {hours}h x {OVERHEAD} overhead)")
    print(f"all-phase actuals:  ${total_spent:.2f} (hard ceiling ${HARD_CEILING_USD:.2f})")

    if actual + projected > cap:
        print(f"FAIL: would put phase {phase} at ${actual + projected:.2f}, over the ${cap:.2f} cap. Not launching.")
        return False, projected
    if total_spent + projected > HARD_CEILING_USD:
        print(f"FAIL: would put total spend at ${total_spent + projected:.2f}, over the ${HARD_CEILING_USD:.2f} hard ceiling. Not launching.")
        return False, projected
    if cap > 0 and (actual + projected) >= 0.8 * cap:
        print(f"WARNING: this run takes phase {phase} to {(actual + projected) / cap:.0%} of its cap. Plan says: stop and re-scope before the NEXT run.")
    print("PASS")
    return True, projected


def write_manifest_stub(args, projected):
    run_id = datetime.datetime.now().strftime(f"{args.module}-%Y%m%d-%H%M%S")
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {"launch_cmd": args.launch, "gpu": args.gpu, "est_hours": args.hours,
              "projected_usd": projected, "seed": args.seed}
    manifest = {
        "run_id": run_id,
        "module": args.module,
        "phase": args.phase,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_revision_sha": None,
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "config": config,
        "seed": args.seed,
        "modal_run_id": None,
        "gpu_type": args.gpu,
        "wall_hours": None,
        "actual_usd": None,
        "artifacts": [],
        "exported_browser_files": [],
        "deck_metrics": {},
        "eval_seed_set": None,
        "notes": "stub written by modal_guard at launch; fill actual_usd + artifacts after the run",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest stub: {run_dir / 'manifest.json'}")
    return run_id


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="check the budget only")
    mode.add_argument("--launch", metavar="MODAL_ARGS", help="check, then `modal run --detach <MODAL_ARGS>`")
    ap.add_argument("--gpu", required=True, help="a10g | a100-40gb | a100-80gb | h100 | t4 | l4 | cpu")
    ap.add_argument("--hours", type=float, required=True, help="estimated wall hours (from the dress rehearsal)")
    ap.add_argument("--cores", type=int, default=1, help="physical cores, only used with --gpu cpu")
    ap.add_argument("--phase", type=int, required=True)
    ap.add_argument("--module", default="shared", help="module name for the manifest stub")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ok, projected = check(args.gpu, args.hours, args.phase, args.cores)
    if not ok:
        sys.exit(1)
    if args.dry_run:
        return
    write_manifest_stub(args, projected)
    cmd = ["modal", "run", "--detach"] + args.launch.split()
    print("launching:", " ".join(cmd))
    sys.exit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
