/* The searches, in the browser: a line-for-line JS mirror of nano_muzero/mcts.py (latent
 * MCTS) and baseline.py's capstone MCTS (true rules + net), plus the Tic-Tac-Toe engine.
 * The 1.8 lab runs BOTH on the same position and lets you inspect every divergence.
 * Deterministic: no Math.random anywhere; noise is injected by the caller when wanted.
 */
import { AlphaZeroJS, MuZeroJS } from "./forward.js";

/* ---------------------------------------------------------------- tic-tac-toe (envs/) */
export const TTT = {
  nActions: 9,
  LINES: [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]],
  initial: () => new Array(9).fill(0),
  legal: (s) => s.reduce((a, c, i) => (c === 0 ? (a.push(i), a) : a), []),
  lineWinner(s) {
    for (const [i, j, k] of this.LINES) if (s[i] !== 0 && s[i] === s[j] && s[i] === s[k]) return s[i];
    return 0;
  },
  isTerminal(s) { return this.lineWinner(s) !== 0 || s.every((c) => c !== 0); },
  winner(s) { return this.lineWinner(s); },
  toPlay: (s) => (s.filter((c) => c !== 0).length % 2 === 0 ? 1 : -1),
  apply(s, a) {
    if (this.isTerminal(s) || s[a] !== 0) throw new Error(`illegal move ${a}`);
    const n = s.slice();
    n[a] = this.toPlay(s);
    return n;
  },
  encode(s) {
    // two planes (mine, theirs) from the to-play view, flattened to 18 floats
    const me = this.toPlay(s), out = new Float64Array(18);
    s.forEach((c, i) => { if (c === me) out[i] = 1; else if (c === -me) out[9 + i] = 1; });
    return out;
  },
};

/* ------------------------------------------------------- latent MCTS (mirror: mcts.py) */
export const MuZeroCfg = (over = {}) => Object.assign({
  nSims: 50, c1: 1.25, c2: 19652, discount: 1.0,
  dirichletAlpha: 0.3, noiseFrac: 0.25,
  useMinmax: true, parentVisitsFromNode: true, skipZeroPrior: false,
}, over);

export const capstoneEquivalent = (nSims) => MuZeroCfg({
  nSims, c1: 1.5, c2: Infinity, useMinmax: false,
  parentVisitsFromNode: false, skipZeroPrior: true,
});

class MinMaxStats {
  constructor() { this.lo = Infinity; this.hi = -Infinity; }
  update(q) { this.lo = Math.min(this.lo, q); this.hi = Math.max(this.hi, q); }
  normalize(q) { return this.hi > this.lo ? (q - this.lo) / (this.hi - this.lo) : q; }
}

function softmax(logits) {
  let m = -Infinity;
  for (const v of logits) m = Math.max(m, v);
  const e = Array.from(logits, (v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

class LatentNode {
  constructor(s, reward, edgePriors) {
    this.s = s; this.reward = reward; this.edgePriors = edgePriors;
    this.n = 0; this.w = 0; this.children = new Map();
  }
  qFromParent(discount) { return this.reward + discount * (-this.w / this.n); }
}

function selectAction(node, cfg, minmax) {
  const parentN = cfg.parentVisitsFromNode
    ? node.n
    : [...node.children.values()].reduce((a, c) => a + c.n, 0);
  const growth = Number.isFinite(cfg.c2)
    ? cfg.c1 + Math.log((parentN + cfg.c2 + 1) / cfg.c2) : cfg.c1;
  let bestA = null, bestScore = -Infinity;
  const actions = [...node.edgePriors.keys()].sort((a, b) => a - b);
  for (const a of actions) {
    const child = node.children.get(a);
    const n = child ? child.n : 0;
    if (cfg.skipZeroPrior && node.edgePriors.get(a) === 0 && n === 0) continue;
    let q = 0;
    if (child && child.n > 0) {
      q = child.qFromParent(cfg.discount);
      if (cfg.useMinmax) q = minmax.normalize(q);
    }
    const score = q + node.edgePriors.get(a) * Math.sqrt(parentN) / (1 + n) * growth;
    if (score > bestScore) { bestA = a; bestScore = score; }
  }
  return bestA;
}

/* model protocol: initial(root) -> {s, logits, v}; recurrent(s, a) -> {s, r, logits, v}.
 * Returns {counts, rootValue, root} -- the root node is exposed so the 1.8 lab's
 * instruments can walk the whole tree (Q values, latents, rewards, per-edge visits). */
export function runMCTS(model, rootInput, legalActions, cfg, noiseFn = null) {
  const ini = model.initial(rootInput);
  const p = softmax(ini.logits);
  const legal = [...legalActions].sort((a, b) => a - b);
  let priors = new Map(legal.map((a) => [a, p[a]]));
  const norm = [...priors.values()].reduce((a, b) => a + b, 0) || 1;
  priors = new Map([...priors].map(([a, pr]) => [a, pr / norm]));
  if (noiseFn) { // caller supplies Dirichlet-ish noise: noiseFn(k) -> array of k weights
    const noise = noiseFn(priors.size);
    let i = 0;
    priors = new Map([...priors].map(([a, pr]) => [a, (1 - cfg.noiseFrac) * pr + cfg.noiseFrac * noise[i++]]));
  }
  const root = new LatentNode(ini.s, 0, priors);
  const minmax = new MinMaxStats();

  for (let sim = 0; sim < cfg.nSims; sim++) {
    let node = root;
    const path = [root];
    let a = selectAction(node, cfg, minmax);
    while (node.children.has(a)) {
      node = node.children.get(a);
      path.push(node);
      a = selectAction(node, cfg, minmax);
    }
    const rec = model.recurrent(node.s, a);
    const pri = softmax(rec.logits);
    const child = new LatentNode(rec.s, rec.r, new Map(pri.map((pr, i) => [i, pr])));
    node.children.set(a, child);
    path.push(child);
    let v = rec.v;
    for (let i = path.length - 1; i >= 0; i--) {
      const nd = path[i];
      nd.n += 1;
      nd.w += v;
      if (nd !== root) minmax.update(nd.qFromParent(cfg.discount));
      v = nd.reward + cfg.discount * (-v);
    }
  }
  const counts = new Array(model.nActions ?? 9).fill(0);
  for (const [a, ch] of root.children) counts[a] = ch.n;
  return { counts, rootValue: root.w / root.n, root };
}

/* Adapters into the model seat */
export class MuZeroModelJS {
  constructor(weightsJson) { this.net = new MuZeroJS(weightsJson); this.nActions = 9; }
  initial(board) {
    const r = this.net.initial(TTT.encode(board));
    return { s: r.s, logits: r.logits, v: r.v };
  }
  recurrent(s, a) { return this.net.recurrent(s, a); }
}

/* the oracle seat (mirror: oracle.py) -- true rules dressed up as h/g/f, for the lab's
 * annotate-overlay and the in-browser oracle demo */
export class OracleModelJS {
  constructor(capstoneWeights) { this.net = new AlphaZeroJS(capstoneWeights); this.nActions = 9; }
  _logits(board) {
    const { logits, v } = this.net.forward(TTT.encode(board));
    const p = softmax(logits.map((l, i) => (board[i] === 0 ? l : -Infinity)));
    return { logits: p.map((x, i) => (board[i] === 0 ? Math.log(x) : -1e9)), v };
  }
  initial(board) { const r = this._logits(board); return { s: board, logits: r.logits, v: r.v }; }
  recurrent(board, a) {
    if (TTT.isTerminal(board) || board[a] !== 0)
      return { s: board, r: 0, logits: new Array(9).fill(0), v: 0 };
    const nxt = TTT.apply(board, a);
    if (TTT.isTerminal(nxt))
      return { s: nxt, r: TTT.winner(nxt) * TTT.toPlay(board), logits: new Array(9).fill(0), v: 0 };
    const r = this._logits(nxt);
    return { s: nxt, r: 0, logits: r.logits, v: r.v };
  }
}

/* --------------------------------------- capstone MCTS (mirror: baseline.py run_mcts) */
export function runCapstoneMCTS(capstoneNet, rootState, nSims, cPuct = 1.5) {
  const netEval = (s) => {
    const { logits, v } = capstoneNet.forward(TTT.encode(s));
    return { p: softmax(logits.map((l, i) => (s[i] === 0 ? l : -Infinity))), v };
  };
  const mkNode = (state, prior) => ({ state, prior, n: 0, w: 0, children: new Map() });
  const expand = (node) => {
    const { p, v } = netEval(node.state);
    for (const a of TTT.legal(node.state)) node.children.set(a, mkNode(TTT.apply(node.state, a), p[a]));
    return v;
  };
  const selectChild = (node) => {
    const total = [...node.children.values()].reduce((a, c) => a + c.n, 0);
    let best = null, bestScore = -Infinity;
    for (const a of [...node.children.keys()].sort((x, y) => x - y)) {
      const ch = node.children.get(a);
      const q = ch.n === 0 ? 0 : -ch.w / ch.n;
      const score = q + cPuct * ch.prior * Math.sqrt(total) / (1 + ch.n);
      if (score > bestScore) { best = ch; bestScore = score; }
    }
    return best;
  };
  const root = mkNode(rootState, 0);
  expand(root);
  for (let i = 0; i < nSims; i++) {
    let node = root;
    const path = [root];
    while (node.children.size) { node = selectChild(node); path.push(node); }
    let v = TTT.isTerminal(node.state)
      ? TTT.winner(node.state) * TTT.toPlay(node.state)
      : expand(node);
    for (let j = path.length - 1; j >= 0; j--) { path[j].n += 1; path[j].w += v; v = -v; }
  }
  const counts = new Array(9).fill(0);
  for (const [a, ch] of root.children) counts[a] = ch.n;
  return { counts, root };
}
