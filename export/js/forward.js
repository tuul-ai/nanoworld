/* Dependency-free JS forward pass for exported nanoworld nets.
 *
 * Contract with shared/nets.py: that file may not gain a primitive without its twin here.
 * Implemented twins: linear + relu/elu/silu/tanh (the mlp family nano_muzero uses).
 * conv_encoder / conv_decoder / ResidualBlock twins land with Module 2's first conv export;
 * until then no conv checkpoint is exported, so the contract holds vacuously.
 *
 * The browser lab runs the learner's own trained nets through THIS file -- the thing in the
 * browser is the thing you trained, guaranteed by export/js/check_golden.mjs.
 */

export function linear(x, W, b) {
  // W.shape = [out, in], W.data row-major (PyTorch nn.Linear convention)
  const [out, inn] = W.shape;
  const y = new Float64Array(out);
  for (let o = 0; o < out; o++) {
    let acc = b ? b.data[o] : 0;
    const row = o * inn;
    for (let i = 0; i < inn; i++) acc += W.data[row + i] * x[i];
    y[o] = acc;
  }
  return y;
}

export const ACTS = {
  relu: (x) => x.map((v) => (v > 0 ? v : 0)),
  elu: (x) => x.map((v) => (v > 0 ? v : Math.exp(v) - 1)),
  silu: (x) => x.map((v) => v / (1 + Math.exp(-v))),
  tanh: (x) => x.map(Math.tanh),
  identity: (x) => x,
};

/* Run an exported shared.nets.mlp: params under `prefix.{0,2,4,...}.weight/bias`,
 * activation between layers, optional output activation -- mirrors nets.py exactly. */
export function mlpForward(x, params, prefix, { hidden = 1, act = "relu", outAct = null } = {}) {
  let h = Float64Array.from(x);
  let idx = 0;
  for (let l = 0; l < hidden; l++) {
    h = ACTS[act](linear(h, params[`${prefix}.${idx}.weight`], params[`${prefix}.${idx}.bias`]));
    idx += 2;
  }
  h = linear(h, params[`${prefix}.${idx}.weight`], params[`${prefix}.${idx}.bias`]);
  if (outAct) h = ACTS[outAct](h);
  return h;
}

export function rescaleLatent(s) {
  // mirror of model.rescale_latent: per-vector min-max to [0,1]
  let lo = Infinity, hi = -Infinity;
  for (const v of s) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const span = Math.max(hi - lo, 1e-8);
  return s.map((v) => (v - lo) / span);
}

export class MuZeroJS {
  /* weightsJson = {config: {obs_dim, latent_dim, hidden, n_actions}, params: {...}} */
  constructor(weightsJson) {
    this.cfg = weightsJson.config;
    this.p = weightsJson.params;
    this.n_actions = this.cfg.n_actions;
  }

  predict(s) {
    // f: trunk (Linear,ReLU,Linear,ReLU) -> policy (Linear) + value (Linear,Tanh)
    const t = mlpForward(s, this.p, "f_trunk", { hidden: 1, outAct: "relu" });
    const logits = mlpForward(t, this.p, "f_policy", { hidden: 0 });
    const v = mlpForward(t, this.p, "f_value", { hidden: 0, outAct: "tanh" })[0];
    return { logits, v };
  }

  initial(obs) {
    // h (Linear,ReLU,Linear) then rescale, then f
    const s = rescaleLatent(mlpForward(obs, this.p, "h", { hidden: 1 }));
    return { s, ...this.predict(s) };
  }

  recurrent(s, a) {
    // g: concat(s, one-hot a) -> core (Linear,ReLU,Linear,ReLU) -> state + reward heads
    const x = new Float64Array(s.length + this.n_actions);
    x.set(s);
    x[s.length + a] = 1;
    const core = mlpForward(x, this.p, "g_core", { hidden: 1, outAct: "relu" });
    const sNext = rescaleLatent(mlpForward(core, this.p, "g_state", { hidden: 0 }));
    const r = mlpForward(core, this.p, "g_reward", { hidden: 0, outAct: "tanh" })[0];
    return { s: sNext, r, ...this.predict(sNext) };
  }
}

export class AlphaZeroJS {
  /* The capstone's two-headed net (baseline.py AlphaZeroNet): the TRUE search's oracle
   * in the 1.8 lab. weightsJson = {config: {obs_dim, hidden, n_actions}, params}. */
  constructor(weightsJson) {
    this.cfg = weightsJson.config;
    this.p = weightsJson.params;
  }

  forward(obs) {
    // trunk (Linear,ReLU,Linear,ReLU) -> policy (Linear) + value (Linear,Tanh)
    const t = mlpForward(obs, this.p, "trunk", { hidden: 1, outAct: "relu" });
    const logits = mlpForward(t, this.p, "policy_head", { hidden: 0 });
    const v = mlpForward(t, this.p, "value_head", { hidden: 0, outAct: "tanh" })[0];
    return { logits, v };
  }
}
