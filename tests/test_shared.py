"""shared/ commons tests: seed reproducibility, ckpt roundtrip, shapes, buffer determinism,
manifest schema validation. All CPU, bit-deterministic (conftest)."""
import json

import numpy as np
import pytest
import torch

from shared import buffer, export, log, nets, seed


def test_seed_reproducibility():
    g1 = seed.seed_all(123)
    a = torch.randn(4, 4)
    r1 = torch.randint(0, 100, (8,), generator=g1)
    g2 = seed.seed_all(123)
    b = torch.randn(4, 4)
    r2 = torch.randint(0, 100, (8,), generator=g2)
    assert torch.equal(a, b)
    assert torch.equal(r1, r2)


def test_pick_device_env_override(monkeypatch):
    monkeypatch.setenv("NANOWORLD_DEVICE", "cpu")
    assert seed.pick_device().type == "cpu"


def test_mlp_shapes_and_grad():
    net = nets.mlp(10, [32, 32], 5)
    x = torch.randn(7, 10)
    y = net(x)
    assert y.shape == (7, 5)
    y.sum().backward()
    assert all(p.grad is not None for p in net.parameters())


def test_conv_roundtrip_shapes():
    enc = nets.conv_encoder(in_ch=3, base=8, depth=4)
    x = torch.randn(2, 3, 64, 64)
    z = enc(x)
    assert z.shape == (2, 64, 4, 4)
    dec = nets.conv_decoder(in_dim=128, out_ch=3, base=8, depth=4)
    img = dec(torch.randn(2, 128))
    assert img.shape == (2, 3, 64, 64)


def test_residual_block_preserves_shape():
    block = nets.ResidualBlock(16)
    x = torch.randn(2, 16, 8, 8)
    assert block(x).shape == x.shape


def test_ckpt_roundtrip(tmp_path):
    net = nets.mlp(4, [8], 2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    log.save_ckpt(path, net, opt, step=42, extra={"note": "hi"})
    net2 = nets.mlp(4, [8], 2)
    opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    payload = log.load_ckpt(path, net2, opt2)
    assert payload["step"] == 42
    assert payload["extra"]["note"] == "hi"
    x = torch.randn(3, 4)
    assert torch.equal(net(x), net2(x))


def test_csv_logger(tmp_path):
    lg = log.CSVLogger(tmp_path, echo_every=100)
    lg.log(step=0, loss=1.0)
    lg.log(step=1, loss=0.5)
    rows = (tmp_path / "metrics.csv").read_text().strip().splitlines()
    assert rows[0] == "step,loss"
    assert len(rows) == 3


def test_replay_buffer_determinism():
    buf = buffer.ReplayBuffer(capacity=50)
    for i in range(60):
        buf.add(obs=np.full(3, i), action=np.array([i % 5]))
    assert len(buf) == 50
    g = torch.Generator().manual_seed(7)
    s1 = buf.sample(16, g)
    g = torch.Generator().manual_seed(7)
    s2 = buf.sample(16, g)
    assert torch.equal(s1["obs"], s2["obs"])
    assert torch.equal(s1["action"], s2["action"])
    assert s1["obs"].shape == (16, 3)


def test_sequence_buffer_windows():
    buf = buffer.SequenceBuffer(capacity_episodes=4, seq_len=10)
    for e in range(3):
        T = 30 + e
        buf.add_episode({"obs": np.arange(T * 2, dtype=np.float32).reshape(T, 2),
                         "action": np.zeros((T, 1))})
    g = torch.Generator().manual_seed(3)
    batch = buf.sample(5, g)
    assert batch["obs"].shape == (5, 10, 2)
    # windows are contiguous: consecutive obs rows differ by the episode stride
    diffs = batch["obs"][:, 1:, 0] - batch["obs"][:, :-1, 0]
    assert torch.allclose(diffs, torch.full_like(diffs, 2.0))


def test_sequence_buffer_rejects_misaligned():
    buf = buffer.SequenceBuffer(capacity_episodes=2, seq_len=5)
    with pytest.raises(ValueError):
        buf.add_episode({"obs": np.zeros((10, 2)), "action": np.zeros((9, 1))})


def test_manifest_schema_validation(tmp_path):
    manifest = {
        "run_id": "test-0", "module": "shared", "config_hash": "x" * 8,
        "seed": 0, "gpu_type": "cpu", "created": "2026-07-02T00:00:00+00:00",
    }
    path = log.write_manifest(tmp_path, manifest)
    assert json.loads(path.read_text())["run_id"] == "test-0"
    bad = {**manifest, "unknown_field": 1}
    jsonschema = pytest.importorskip("jsonschema")
    with pytest.raises(jsonschema.ValidationError):
        log.write_manifest(tmp_path, bad)


def test_export_json_and_golden(tmp_path):
    net = nets.mlp(4, [8], 2)
    wpath = export.export_json(net.state_dict(), tmp_path / "w.json")
    payload = json.loads(wpath.read_text())
    assert payload["0.weight"]["shape"] == [8, 4]
    gpath = export.dump_golden("testnet", net, torch.randn(1, 4), out_dir=tmp_path)
    golden = json.loads(gpath.read_text())
    assert golden["output"]["shape"] == [1, 2]
    out = net(torch.tensor(golden["input"]["data"]).reshape(golden["input"]["shape"]))
    assert np.allclose(out.detach().numpy().flatten(), golden["output"]["data"], atol=1e-6)
