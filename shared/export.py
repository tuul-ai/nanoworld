"""Browser export: ONNX for onnxruntime-web, JSON weights for the hand-rolled JS path,
plus the golden-vector dump that is the actual correctness guarantee (checked in node)."""
import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "export" / "golden"


def export_onnx(model: torch.nn.Module, sample_input, path: Path):
    """Static shapes, batch=1, fp32, no loops inside the graph (recurrent state crosses as I/O)."""
    model.eval()
    torch.onnx.export(model, sample_input, str(path), opset_version=17,
                      input_names=["input"], output_names=["output"])
    return Path(path)


def export_json(state_dict: dict, path: Path):
    """Weights as nested lists + shape manifest for export/js/forward.js."""
    payload = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().float()
        payload[name] = {"shape": list(t.shape), "data": t.flatten().tolist()}
    Path(path).write_text(json.dumps(payload))
    return Path(path)


def dump_golden(name: str, model: torch.nn.Module, sample_input, out_dir: Path = None):
    """Fixed input + PyTorch output; node's check_golden.mjs asserts the JS/ONNX path matches."""
    out_dir = Path(out_dir or GOLDEN_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        out = model(sample_input)
    golden = {
        "input": {"shape": list(sample_input.shape), "data": sample_input.flatten().tolist()},
        "output": {"shape": list(out.shape), "data": out.flatten().tolist()},
        "tolerance": 1e-5,
    }
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(golden))
    return path
