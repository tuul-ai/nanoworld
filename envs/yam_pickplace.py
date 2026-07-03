"""YAM single-arm MuJoCo pick-and-place: the nanoworld sim environment (worldmodels course).

Vendored and adapted from the tuul-dev/caferacer snapshot (caferacer/sim.py, plus R_to_quat from
caferacer/geom.py, both inlined; no caferacer import). The robot is the molmoact2 sim-eval YAM MJCF
(vendored under envs/assets/yam/); the scene (wood/white table, cardboard bin, bar object, topdown +
wrist cameras) is assembled at runtime via MjSpec, and the base is offset to ROBOT_POS so a fixed task
geometry (zone/bin/table/grasp) holds. This module owns the sim: scene construction, control, cameras,
deterministic DLS IK (ik_site), the scripted oriented-bar grasp teacher (run_episode), the
grasp_feasible gate, and gated pose sampling (sample_pose).

The physics constants are the caferacer-tuned, mujoco 3.10.0 re-validated numbers (1 kHz timestep,
Newton solver, elliptic cones, impratio 30, ARM_DAMPING 2.0, object slide friction 1.0, firm kp-1000
force-limited gripper). Do not change them without re-running the gated-pose teacher suite.
Dropped relative to the source: the film camera path, the affordance/glyph pixel-capture machinery,
and the cinematic traj hook. The scripted teacher logic (drive phases, close_ramp, cartesian carry,
honest landed predicate) is identical.

Smoke (headless offscreen render, works on macOS via the default CGL path):
  venv/bin/python -m envs.yam_pickplace
Tests (no GL needed, dummy renderers):
  venv/bin/python -m pytest envs/test_yam_pickplace.py -q
"""
import os
import math
import numpy as np
import mujoco

# ---- vendored YAM asset (molmoact2 sim-eval MJCF + meshes, committed under envs/assets/yam/) ----
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
YAM_XML = os.path.join(ASSET_DIR, "yam", "bimanual_yam_linear_flattened.xml")

# ---- task geometry (mirrors the validated Genesis design; single-sourced numbers) ----
ROBOT_POS = (-0.65, 0.0, 0.01)                 # offset the YAM base so the zone/bin/table transfer from Genesis
OBJECT = os.environ.get("OBJECT", "bar").lower()    # 'bar' (oriented-grasp) | 'lego' (square)
LEGO = (0.055, 0.035, 0.025)
BAR = (0.072, 0.028, 0.025)                    # elongated bar (l x w x h): orientation is a real grasp DOF
OBJ_SIZE = BAR if OBJECT == "bar" else LEGO
GZ = 0.0125                                     # object center height (half of 0.025)
ZONE = dict(x=(-0.36, -0.20), y=((0.12, 0.26) if OBJECT == "bar" else (0.06, 0.26)))
BIN_CENTER = (-0.18, 0.02)
BIN_INNER_HALF = 0.06
REJECT = 0.085
LAND_TOL = 0.06
LIFT_OK = 0.06
HOVER_Z, LIFT_Z = 0.11, 0.14
WS = (-0.28, 0.15)                              # workspace center (cameras aim here)

ARM_DAMPING = float(os.environ.get("ARM_DAMPING", "2.0"))   # joint damping on the 6 arm DOFs: the YAM asset
#  ships with dof_damping=0, so the underdamped position servos ring (j3 ~20% overshoot) -> visible "springy"
#  wobble. ~2.0 critically damps it (0% overshoot, no ring) without making the teacher's timed moves sluggish.
OBJ_SLIDE_FRIC = float(os.environ.get("OBJ_FRICTION", "1.0"))   # object slide friction. 1.0 = realistic rubber-
#  pad coefficient; grasp success is identical 96% from 2.5 down to 0.8 (impratio=30 does the anti-slip work),
#  so 2.5 was gratuitously sticky (sim-to-real-optimistic). 1.0 is honest with margin.
FINGER_OPEN = 0.0475                            # gripper ctrl: -0.0475 open, 0 closed
GRASP_SITE = "left_grasp_site"                  # model's own grasp site = local (0,0,0.1347) on left_link_6
LEFT_ARM = [f"left_joint{i}" for i in range(1, 7)]
LEFT_FINGERS = ["left_gripper_left_tip", "left_gripper_right_tip"]
RES = (256, 256)                                # (W, H)

# cameras: standard rig = top-down + wrist
TOPDOWN = dict(pos=(WS[0], WS[1], 0.62), fovy=52)
WRIST = dict(pos_local=(0.06, 0.0, 0.035), look_local=(0.0, 0.0, 0.12), fovy=58)   # on left_link_6


def R_to_quat(R):
    """3x3 rotation -> quaternion (w,x,y,z), normalized. Shepperd's method (numerically stable).
    Inlined from caferacer/geom.py (pure numpy)."""
    R = np.asarray(R, np.float64)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], np.float64)
    return (q / np.linalg.norm(q)).astype(np.float32)


def _add_box(body, name, size, pos, rgba, contype=1, conaffinity=1, group=0, density=1000.0):
    g = body.add_geom()
    g.name = name; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [size[0] / 2, size[1] / 2, size[2] / 2]; g.pos = pos
    g.rgba = rgba; g.contype = contype; g.conaffinity = conaffinity; g.group = group; g.density = density
    return g


def _open_box(spec, center, inner_half, height=0.05, wall=0.008, white=False):
    """Cardboard open-top bin = floor + 4 walls (fixed). Mirrors the Genesis build_open_box look."""
    bx, by = center; outer = inner_half + wall
    card = [0.58, 0.42, 0.25, 1.0]
    b = spec.worldbody.add_body(); b.name = "bin"; b.pos = [bx, by, 0.0]
    _add_box(b, "bin_floor", (2 * outer, 2 * outer, wall), [0, 0, wall / 2], [0.72, 0.61, 0.43, 1.0])
    wz = wall + height / 2
    for sx in (-1, 1):
        _add_box(b, f"bin_wx{sx}", (wall, 2 * outer, height), [sx * (inner_half + wall / 2), 0, wz], card)
    for sy in (-1, 1):
        _add_box(b, f"bin_wy{sy}", (2 * inner_half, wall, height), [0, sy * (inner_half + wall / 2), wz], card)


def _look_quat(eye, target, up=(0, 0, 1)):
    """Camera quat (wxyz) for a camera at `eye` looking at `target`. MuJoCo cameras look along their -Z
    (z points backward), x=right, y=up -> R columns [right, up, -forward]."""
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, float)); right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    R = np.stack([right, up2, -fwd], axis=1)
    return list(R_to_quat(R))


def build(object_kind=None, table_style=None):
    """Assemble the scene (MjSpec): YAM left arm + table + bin + object + topdown/wrist cameras. Returns a
    dict of handles {model, data, ids...}. object_kind/table_style default to env (OBJECT/TABLE_STYLE)."""
    assert YAM_XML and os.path.exists(YAM_XML), f"YAM xml not found: {YAM_XML}"
    kind = (object_kind or OBJECT).lower()
    obj_size = BAR if kind == "bar" else LEGO
    white = (table_style or os.environ.get("TABLE_STYLE", "white")).lower() == "white"

    spec = mujoco.MjSpec.from_file(YAM_XML)
    # grasp-contact stability (match the Genesis recipe that made real friction hold: fine substepping +
    # many solver iterations + implicit integrator). MuJoCo has no substeps, so use a fine timestep instead.
    spec.option.timestep = 0.001                # 1 kHz (~Genesis effective dt with substeps=8)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 100
    # anti-slip recipe (MuJoCo docs: "elliptic cones, large impratio, and the Newton algorithm with very
    # small tolerance" is the canonical fix for contact slip).
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 30.0
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.tolerance = 1e-10
    # offset the robot base so the Genesis task geometry (zone/bin/table) applies unchanged
    base = spec.body("bimanual_base"); base.pos = list(ROBOT_POS)
    # firmer LEFT gripper (kp 100->1000) so the closed grip resists the carry's lateral accel; the close is
    # RAMPED (not snapped) in run_episode so this stiffness grips firmly instead of ejecting the bar.
    # robosuite Panda-style firm grip: kp=1000 position servo, force-limited (a soft low-kp servo gets pried
    # open by the carried object's lateral inertia).
    for an in LEFT_FINGERS:
        act = spec.actuator(an)
        gp = np.zeros(10); gp[0] = 1000.0; bp = np.zeros(10); bp[1] = -1000.0; bp[2] = -40.0
        act.gainprm = gp; act.biasprm = bp
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE; act.forcerange = [-20.0, 20.0]

    # lighting (the asset ships dim) -> bright studio look + a key light with soft shadows
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    key = spec.worldbody.add_light()
    key.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    key.pos = [-0.25, -0.1, 1.3]; key.dir = [0.1, 0.15, -1.0]
    key.diffuse = [0.65, 0.65, 0.65]; key.specular = [0.25, 0.25, 0.25]; key.castshadow = True
    fill = spec.worldbody.add_light()
    fill.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    fill.pos = [0.2, 0.5, 1.0]; fill.dir = [-0.1, -0.3, -1.0]
    fill.diffuse = [0.3, 0.3, 0.3]; fill.castshadow = False

    # floor (dark studio backdrop) + table (white, or wood-ish)
    floor_rgba = [0.07, 0.07, 0.09, 1.0] if white else [0.15, 0.15, 0.17, 1.0]
    _add_box(spec.worldbody, "floor", (4.0, 4.0, 0.02), [0, 0, -0.09], floor_rgba)
    table_rgba = [0.92, 0.92, 0.93, 1.0] if white else [0.55, 0.38, 0.22, 1.0]
    _add_box(spec.worldbody, "table", (1.04, 0.78, 0.05), [-0.25, 0.0, -0.025], table_rgba)

    _open_box(spec, BIN_CENTER, BIN_INNER_HALF, white=white)

    # object body: box collider (stable grasp) + lego-mesh visual on ONE free body
    obj = spec.worldbody.add_body(); obj.name = "obj"; obj.pos = [WS[0], WS[1], GZ]
    obj.add_freejoint()
    col = _add_box(obj, "obj_col", obj_size, [0, 0, 0],
                   [0.82, 0.12, 0.10, 0.0 if kind == "bar" else 1.0], density=496.0)  # ~25 g; hidden for bar
    col.condim = 4                              # torsional friction -> docs: "substantially improves grasping"
    col.friction = [OBJ_SLIDE_FRIC, 0.05, 0.001]   # slide / torsional / rolling (1.0 = realistic, see const)
    if kind == "bar":
        lego_obj = os.path.join(ASSET_DIR, "lego", "lego_brick.obj")
        if os.path.exists(lego_obj):
            mesh = spec.add_mesh(); mesh.name = "lego"; mesh.file = lego_obj
            vg = obj.add_geom(); vg.name = "obj_vis"; vg.type = mujoco.mjtGeom.mjGEOM_MESH
            vg.meshname = "lego"; vg.rgba = [0.82, 0.12, 0.10, 1.0]
            vg.contype = 0; vg.conaffinity = 0; vg.group = 2; vg.density = 0.0
        else:
            col.rgba = [0.82, 0.12, 0.10, 1.0]

    # cameras (mode fixed; quats orient them). topdown world-fixed; wrist on left_link_6.
    ct = spec.worldbody.add_camera(); ct.name = "topdown"; ct.pos = list(TOPDOWN["pos"])
    ct.fovy = TOPDOWN["fovy"]; ct.quat = _look_quat(TOPDOWN["pos"], (WS[0], WS[1], 0.0), up=(0, 1, 0))
    ee = spec.body("left_link_6")
    cw = ee.add_camera(); cw.name = "wrist"; cw.pos = list(WRIST["pos_local"]); cw.fovy = WRIST["fovy"]
    cw.quat = _look_quat(WRIST["pos_local"], WRIST["look_local"], up=(0, 0, 1))

    model = spec.compile()
    for jn in LEFT_ARM:                          # damp the underdamped arm servos (asset ships dof_damping=0 ->
        dofadr = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)]
        model.dof_damping[dofadr] = ARM_DAMPING  # position servos ring ~20% -> visible "springy" wobble)
    data = mujoco.MjData(model)

    def jid(n, obj=mujoco.mjtObj.mjOBJ_JOINT): return mujoco.mj_name2id(model, obj, n)
    H = dict(spec=spec, model=model, data=data, kind=kind,
             larm_qadr=[model.jnt_qposadr[jid(n)] for n in LEFT_ARM],
             larm_dofadr=[model.jnt_dofadr[jid(n)] for n in LEFT_ARM],
             larm_act=[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in LEFT_ARM],
             finger_act=[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in LEFT_FINGERS],
             finger_qadr=[model.jnt_qposadr[jid(n)] for n in ("left_left_finger", "left_right_finger")],
             grasp_site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRASP_SITE),
             ee_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_link_6"),
             obj_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj"),
             obj_qadr=model.jnt_qposadr[model.body_jntadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj")]],
             larm_range=[model.jnt_range[jid(n)].copy() for n in LEFT_ARM],
             finger_body=[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                          for n in ("left_link_left_finger", "left_link_right_finger")],
             cam={n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in ("topdown", "wrist")})
    # home: all arm joints 0, grippers open
    home_grip(H)
    mujoco.mj_forward(model, data)
    return H


# ----------------------------------------------------------------------------- kinematics / IK
def get_arm(H):
    return np.array([H["data"].qpos[a] for a in H["larm_qadr"]], np.float32)


def _set_arm(H, q_left):
    d = H["data"]
    for a, q in zip(H["larm_qadr"], q_left):
        d.qpos[a] = q
    mujoco.mj_kinematics(H["model"], d); mujoco.mj_comPos(H["model"], d)


def grasp_pose(H):
    """grasp-site world (pos (3,), R (3,3))."""
    d = H["data"]; sid = H["grasp_site"]
    return d.site_xpos[sid].copy().astype(np.float32), d.site_xmat[sid].reshape(3, 3).copy().astype(np.float32)


def ik_site(H, tpos, tquat=None, seed_left=None, iters=200, damping=0.06, pos_tol=4e-4, rot_tol=2e-3):
    """Deterministic damped-least-squares IK of the grasp SITE to tpos[, tquat], moving ONLY the left arm
    (right arm stays parked). tquat=None -> position-only (the natural pose). Returns the 6 left-arm angles."""
    m, d = H["model"], H["data"]; sid = H["grasp_site"]
    qsave = d.qpos.copy()                        # IK mutates qpos (FK probes); restore so it is side-effect-free
    if seed_left is not None:                    # -- else a mid-motion IK call would TELEPORT the gripper off
        _set_arm(H, seed_left)                   #    the grasped object (free body doesn't follow a teleport)
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv)); cols = H["larm_dofadr"]
    tpos = np.asarray(tpos, float)
    er = np.zeros(3); cq = np.zeros(4); cq_conj = np.zeros(4); eq = np.zeros(4)
    tq = np.asarray(tquat, float) if tquat is not None else None
    for _ in range(iters):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        epos = tpos - d.site_xpos[sid]
        if tq is not None:
            # WORLD-frame orientation error = target (x) conj(current) -> rotvel, matching mj_jacSite's jacr
            mujoco.mju_mat2Quat(cq, d.site_xmat[sid]); mujoco.mju_negQuat(cq_conj, cq)
            mujoco.mju_mulQuat(eq, tq, cq_conj); mujoco.mju_quat2Vel(er, eq, 1.0)
            err = np.concatenate([epos, er])
        else:
            err = epos
        if np.linalg.norm(epos) < pos_tol and (tquat is None or np.linalg.norm(er) < rot_tol):
            break
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        J = jacp[:, cols] if tquat is None else np.vstack([jacp[:, cols], jacr[:, cols]])
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(J.shape[0]), err)
        n = np.max(np.abs(dq))                                         # clamp step so large rolls don't diverge
        if n > 0.1:
            dq *= 0.1 / n
        for k, a in enumerate(H["larm_qadr"]):
            lo, hi = H["larm_range"][k]
            d.qpos[a] = min(max(d.qpos[a] + dq[k], lo), hi)
    sol = get_arm(H)
    d.qpos[:] = qsave; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)   # restore live physical state
    return sol


def grasp_quat_oriented(H, cx, cy, yaw):
    """Oriented vertical-jaw grasp quat (wxyz) for a bar at (cx,cy,yaw): position-only natural pose, then
    roll about WORLD-Z so the jaw (grasp-site col0) straddles the bar's short axis (validated on Genesis).
    Returns (quat_wxyz, q_nat). Restores the arm to HOME after probing."""
    q_nat = ik_site(H, [cx, cy, GZ], seed_left=np.zeros(6))           # deterministic, position-only
    _set_arm(H, q_nat); _, R_nat = grasp_pose(H)                      # set arm to read its grasp-site frame
    b = np.array([-math.sin(yaw), math.cos(yaw), 0.0])               # bar width (jaw target)
    # MuJoCo grasp-site frame: the jaw (finger closing-travel) axis is col1 (verified in mj_diag_grasp)
    dphi = math.atan2(b[1], b[0]) - math.atan2(R_nat[1, 1], R_nat[0, 1])
    dphi = (dphi + math.pi) % (2 * math.pi) - math.pi
    if abs(dphi) > math.pi / 2:
        dphi -= math.copysign(math.pi, dphi)
    c_, s_ = math.cos(dphi), math.sin(dphi)
    Rz = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]])
    R_t = (Rz @ R_nat)
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R_t.reshape(9))
    _set_arm(H, np.zeros(6))                                          # restore HOME
    return q.astype(np.float32), q_nat


def home_grip(H):
    """Open the left gripper via ctrl (and the right parked at home)."""
    H["data"].ctrl[H["finger_act"]] = -FINGER_OPEN


def set_object(H, x, y, yaw=0.0):
    """Place the object free body at (x,y,GZ) with yaw about z (quat wxyz), zero velocity."""
    m, d = H["model"], H["data"]; a = H["obj_qadr"]
    d.qpos[a:a + 3] = [x, y, GZ]
    d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    d.qvel[H["model"].jnt_dofadr[H["model"].body_jntadr[H["obj_body"]]]:][:6] = 0.0
    mujoco.mj_forward(m, d)


def render(H, cam, res=RES, renderer=None):
    """Render an RGB frame (H,W,3 uint8) from a named camera."""
    m, d = H["model"], H["data"]
    own = renderer is None
    r = renderer or mujoco.Renderer(m, height=res[1], width=res[0])
    r.update_scene(d, camera=cam)
    img = r.render().astype(np.uint8)
    if own:
        r.close()
    return np.ascontiguousarray(img)


def left_state7(H):
    """7-D command: 6 left-arm joints (deg) + gripper openness [0,1]."""
    d = H["data"]
    arm = np.degrees([d.qpos[a] for a in H["larm_qadr"]])
    openness = float(np.clip(-d.qpos[H["finger_qadr"][0]] / FINGER_OPEN, 0.0, 1.0))
    return np.concatenate([arm, [openness]]).astype(np.float32)


# ----------------------------------------------------------------------------- scripted episode (teacher)
def _ctrl_left(H, q_left, openness):
    d = H["data"]
    d.ctrl[H["larm_act"]] = q_left
    d.ctrl[H["finger_act"]] = -FINGER_OPEN * float(openness)


def ctrl_from_cmd7(H, cmd7):
    """Drive the position servos from a 7-D command [6 joints deg, gripper openness]; right arm parked."""
    _ctrl_left(H, np.radians(np.asarray(cmd7[:6], float)), float(cmd7[6]))


def grasp_feasible(H, x, y, yaw, max_resid=0.006):
    """Is the scripted oriented grasp at (x,y,yaw) actually executable? Two collision-blind-IK failure modes
    the teacher hits near the workspace/bin edge: (1) the 6-DOF arm can't reach the oriented target (IK clamps
    to joint limits -> large residual), (2) the open gripper, at the grasp config, PENETRATES the bin wall
    (the finger arc overlaps the bin footprint -> the descent jams short and never grips). Returns (ok, resid).
    Side-effect-free (restores qpos)."""
    m, d = H["model"], H["data"]
    qsave = d.qpos.copy()
    quat_t, q_seed = grasp_quat_oriented(H, x, y, yaw)
    q_grasp = ik_site(H, [x, y, GZ], tquat=quat_t, seed_left=q_seed)
    _set_arm(H, q_grasp); c, _ = grasp_pose(H)
    resid = float(np.linalg.norm(c - np.array([x, y, GZ], np.float32)))
    set_object(H, x, y, yaw)

    def _arm_hits_bin():                                 # any left-arm/finger geom contacting the bin?
        for i in range(d.ncon):
            n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom1]) or ""
            n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[d.contact[i].geom2]) or ""
            if ("bin" in (n1, n2)) and (n1.startswith("left_") or n2.startswith("left_")):
                return True
        return False

    # sweep the lower descent path (open gripper): a finger can clip the bin MID-descent even when the final
    # grasp config is clear, so check several heights, not just the endpoint.
    collides = False; seed = q_grasp
    for dz in (0.0, 0.03, 0.06):
        seed = ik_site(H, [x, y, GZ + dz], tquat=quat_t, seed_left=seed)
        _set_arm(H, seed)
        for a in H["finger_qadr"]:
            d.qpos[a] = -FINGER_OPEN                      # open gripper = the descend-approach config
        mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)
        if _arm_hits_bin():
            collides = True; break
    d.qpos[:] = qsave; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    return (resid < max_resid and not collides), resid


def sample_pose(rng, H=None, max_resid=0.006):
    """-> ((x,y), yaw). Bar yaw in [0,pi) (180-symmetric). Rejects spawns within REJECT of the bin. If H is
    given, ALSO rejects poses where the scripted grasp is infeasible (unreachable IK OR the gripper hits the
    bin wall at the grasp config) -- so train + eval draw only from cleanly-graspable poses."""
    x = y = 0.0; yaw = 0.0
    for _ in range(300):
        x = rng.uniform(*ZONE["x"]); y = rng.uniform(*ZONE["y"])
        if math.hypot(x - BIN_CENTER[0], y - BIN_CENTER[1]) < REJECT:
            continue
        yaw = float(rng.uniform(0.0, math.pi)) if OBJECT == "bar" else 0.0
        if H is None or grasp_feasible(H, x, y, yaw, max_resid)[0]:
            return (float(x), float(y)), yaw
    return (float(x), float(y)), yaw   # fallback: last sample (shouldn't happen with a sane reachable zone)


def run_episode(H, cube_xy, yaw=0.0, every=8, renderers=None):
    """Scripted oriented pick-place of the bar at cube_xy onto the bin. Captures clean topdown+wrist frames
    plus the 7-D state every `every` steps. Returns (success, frames, info).
    renderers = {'topdown':Renderer,'wrist':Renderer} (reused for speed; injectable, so tests pass dummies)."""
    m, d = H["model"], H["data"]
    cx0, cy0 = cube_xy; bx, by = BIN_CENTER
    sc = max(1, round(0.004 / m.opt.timestep))   # scale phase step counts to the timestep (same wall-time motion)
    every = every * sc
    own = renderers is None
    if own:
        renderers = {c: mujoco.Renderer(m, height=RES[1], width=RES[0]) for c in ("topdown", "wrist")}

    # reset arm to HOME, place object, settle
    mujoco.mj_resetData(m, d)
    _ctrl_left(H, np.zeros(6), 1.0); d.ctrl[[a for a in range(m.nu) if a not in H["larm_act"] and a not in H["finger_act"]]] = 0.0
    set_object(H, cx0, cy0, yaw)
    for _ in range(60):
        mujoco.mj_step(m, d)

    # deterministic oriented grasp config (shared with eval's oracle)
    quat_t, q_seed = grasp_quat_oriented(H, cx0, cy0, yaw)
    q_grasp = ik_site(H, [cx0, cy0, GZ], tquat=quat_t, seed_left=q_seed)
    q_hover = ik_site(H, [cx0, cy0, GZ + HOVER_Z], tquat=quat_t, seed_left=q_grasp)
    q_lift = ik_site(H, [cx0, cy0, GZ + LIFT_Z], tquat=quat_t, seed_left=q_grasp)
    _set_arm(H, q_grasp); grasp_c, grasp_R = grasp_pose(H)
    ik_resid = float(np.linalg.norm(grasp_c - np.array([cx0, cy0, GZ], np.float32)))  # >tol => unreachable pose
    _set_arm(H, np.zeros(6)); mujoco.mj_forward(m, d)

    frames, ctr, zmax, grasp_dist = [], [0], [GZ], [float("nan")]

    def render(cam):
        renderers[cam].update_scene(d, camera=cam)
        return np.ascontiguousarray(renderers[cam].render().astype(np.uint8))

    def capture():
        frames.append(dict(state=left_state7(H), images={"topdown": render("topdown"), "wrist": render("wrist")}))

    def _step():
        mujoco.mj_step(m, d); ctr[0] += 1
        zmax[0] = max(zmax[0], d.xpos[H["obj_body"]][2])
        if ctr[0] % every == 0:
            capture()

    def drive(q_left, openness, n):
        _ctrl_left(H, q_left, openness)
        for _ in range(n * sc):
            _step()

    def close_ramp(q_left, n):              # ramp openness 1->0 gently so a firm grip doesn't snap-eject the bar
        for s in range(1, n * sc + 1):
            _ctrl_left(H, q_left, 1.0 - s / (n * sc)); _step()

    drive(q_hover, 1.0, 110)                    # move above
    drive(q_grasp, 1.0, 120)                    # descend (straight down in grasp-site pos)
    z0 = d.xpos[H["obj_body"]][2]
    close_ramp(q_grasp, 80); drive(q_grasp, 0.0, 80)   # gentle ramp close + settle
    grasp_dist[0] = float(np.linalg.norm(d.site_xpos[H["grasp_site"]] - d.xpos[H["obj_body"]]))  # ACHIEVED grip
    drive(q_lift, 0.0, 120)                     # lift straight up (oriented)
    # CARTESIAN carry: interpolate the EE POSITION, re-solving IK at quat_t each waypoint, so the orientation
    # tracks quat_t closely (servo lag leaves ~6deg drift by the last waypoint -- fine; a joint-space glide
    # instead tilts the jaws ~tens of deg mid-path and drops the bar).
    def cart_to(p_to, openness, n_way=24, steps=4, seed=None):
        p0 = grasp_pose(H)[0].copy()
        for s in range(1, n_way + 1):
            p = p0 + (np.asarray(p_to, float) - p0) * (s / n_way)
            drive(ik_site(H, p, tquat=quat_t, seed_left=get_arm(H)), openness, steps)
    cart_to([bx, by, GZ + LIFT_Z], 0.0)                           # carry over the bin
    cart_to([bx, by, GZ + 0.07], 0.0, n_way=16)                   # lower into the bin
    drive(get_arm(H), 1.0, 60)                                    # open (release)
    drive(ik_site(H, [bx, by, GZ + LIFT_Z], tquat=quat_t, seed_left=get_arm(H)), 1.0, 50)    # retract

    final = d.xpos[H["obj_body"]].copy()
    long_axis_z = abs(float(d.xmat[H["obj_body"]].reshape(3, 3)[2, 0]))   # |world-z comp of bar's long (local-x) axis|
    lifted = zmax[0] > z0 + LIFT_OK
    # honest landed: COM over the bin AND settled inside (below the rim, not perched/flung) AND lying flat
    # (long axis within ~45deg of horizontal -> reject a bar standing on its end inside the bin).
    in_bin = math.hypot(final[0] - bx, final[1] - by) < LAND_TOL and 0.0 < final[2] < 0.06
    flat = long_axis_z < 0.707
    landed = bool(in_bin and flat)
    if own:
        for r in renderers.values():
            r.close()
    info = dict(cube_xy=[float(cx0), float(cy0)], yaw=float(yaw),
                lifted=bool(lifted), landed=landed, grasp_c=grasp_c, grasp_R=grasp_R,
                ik_resid=ik_resid, grasp_dist=grasp_dist[0], tilt=long_axis_z)
    return bool(lifted and landed), frames, info


def main():
    """Smoke: build, sample one gated pose, run the scripted teacher with REAL offscreen renderers."""
    H = build()
    m = H["model"]
    print(f"[yam] built: nq={m.nq} nbody={m.nbody} ncam={m.ncam} kind={H['kind']}", flush=True)
    rng = np.random.default_rng(1000)
    (x, y), yaw = sample_pose(rng, H)
    print(f"[yam] gated pose: ({x:+.3f},{y:+.3f}) yaw={math.degrees(yaw):.1f}deg", flush=True)
    ok, frames, info = run_episode(H, (x, y), yaw)
    print(f"[yam] success={ok} frames={len(frames)} ik_resid={info['ik_resid']*1000:.2f}mm "
          f"grasp_dist={info['grasp_dist']*1000:.1f}mm", flush=True)


if __name__ == "__main__":
    main()
