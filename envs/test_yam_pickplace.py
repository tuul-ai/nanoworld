"""Tests for envs/yam_pickplace.py: physics config, determinism, and the scripted-grasp gate.

No GL context needed: episodes run with dummy renderer objects (the renderers dict is injectable),
so everything here is pure physics + kinematics. Episodes are ~3,180 steps each, so the full file
takes a few minutes. Expectations match the mujoco 3.10.0 re-validation
(tuul-dev/courses/worldmodels/spikes/mj310-revalidation.md): gated eval seeds succeed 50/50 there,
so seeds 1000-1004 must all pass here.
"""
import math

import mujoco
import numpy as np

from envs import yam_pickplace as yp


class DummyRenderer:
    """Stands in for mujoco.Renderer so run_episode needs no GL context."""

    def update_scene(self, data, camera=None):
        pass

    def render(self):
        return np.zeros((yp.RES[1], yp.RES[0], 3), dtype=np.uint8)

    def close(self):
        pass


def _dummy_renderers():
    return {"topdown": DummyRenderer(), "wrist": DummyRenderer()}


def test_physics_config():
    """The tuned caferacer physics constants must survive the port exactly."""
    H = yp.build()
    opt = H["model"].opt
    assert opt.timestep == 0.001
    assert opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert opt.impratio == 30.0
    # the post-compile overrides
    for dofadr in H["larm_dofadr"]:
        assert H["model"].dof_damping[dofadr] == yp.ARM_DAMPING == 2.0
    obj_geom = mujoco.mj_name2id(H["model"], mujoco.mjtObj.mjOBJ_GEOM, "obj_col")
    assert H["model"].geom_friction[obj_geom][0] == 1.0


def test_determinism():
    """Two fresh builds, same gated pose (seed 1000): identical final qpos, success both times."""
    finals, succ = [], []
    pose = None
    for _ in range(2):
        H = yp.build()
        rng = np.random.default_rng(1000)
        (x, y), yaw = yp.sample_pose(rng, H)
        if pose is None:
            pose = ((x, y), yaw)
        else:
            assert pose == ((x, y), yaw), "sample_pose is not deterministic under a fixed seed"
        ok, frames, info = yp.run_episode(H, (x, y), yaw, renderers=_dummy_renderers())
        succ.append(ok)
        finals.append(H["data"].qpos.copy())
        assert len(frames) > 0 and frames[0]["state"].shape == (7,)
    assert succ == [True, True]
    assert np.array_equal(finals[0], finals[1])


def test_scripted_grasp_gate():
    """Seeds 1000-1004: every gated pose must be picked and placed by the scripted teacher."""
    H = yp.build()
    renderers = _dummy_renderers()
    results = []
    for seed in range(1000, 1005):
        rng = np.random.default_rng(seed)
        (x, y), yaw = yp.sample_pose(rng, H)
        ok, resid = yp.grasp_feasible(H, x, y, yaw)
        assert ok, f"seed {seed}: sample_pose returned an infeasible pose (resid {resid*1000:.2f}mm)"
        success, _, info = yp.run_episode(H, (x, y), yaw, renderers=renderers)
        results.append((seed, (x, y), math.degrees(yaw), success, info["lifted"], info["landed"]))
    failures = [r for r in results if not r[3]]
    assert not failures, f"scripted teacher failed on gated poses: {failures}"
