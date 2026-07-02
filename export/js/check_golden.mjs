/* Golden-vector check: the JS forward pass must reproduce PyTorch's outputs exactly
 * (within tolerance) on fixed inputs. Runs in CI on every push.
 *
 *   node export/js/check_golden.mjs muzero     # one module
 *   node export/js/check_golden.mjs --all      # every export/golden/<name>_golden.json
 */
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { MuZeroJS } from "./forward.js";

const GOLDEN_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "golden");

function maxAbsDiff(a, b) {
  if (a.length !== b.length) return Infinity;
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

function checkMuzero(goldenFile) {
  const g = JSON.parse(readFileSync(join(GOLDEN_DIR, goldenFile)));
  const weights = JSON.parse(readFileSync(join(GOLDEN_DIR, g.weights_file)));
  const net = new MuZeroJS(weights);
  const tol = g.tolerance;

  const ini = net.initial(g.obs);
  const rec = net.recurrent(Float64Array.from(g.initial.s), g.action);
  const checks = [
    ["initial.s", maxAbsDiff(ini.s, g.initial.s)],
    ["initial.logits", maxAbsDiff(ini.logits, g.initial.logits)],
    ["initial.v", Math.abs(ini.v - g.initial.v)],
    ["recurrent.s", maxAbsDiff(rec.s, g.recurrent.s)],
    ["recurrent.logits", maxAbsDiff(rec.logits, g.recurrent.logits)],
    ["recurrent.r", Math.abs(rec.r - g.recurrent.r)],
    ["recurrent.v", Math.abs(rec.v - g.recurrent.v)],
  ];
  let ok = true;
  for (const [name, diff] of checks) {
    const pass = diff <= tol;
    ok &&= pass;
    console.log(`  ${goldenFile} ${name}: max|diff| = ${diff.toExponential(2)} ${pass ? "PASS" : "FAIL"}`);
  }
  return ok;
}

const arg = process.argv[2];
if (!arg) {
  console.error("usage: node check_golden.mjs <module>|--all");
  process.exit(2);
}
const files = arg === "--all"
  ? readdirSync(GOLDEN_DIR).filter((f) => f.endsWith("_golden.json"))
  : [`${arg}_golden.json`];
if (files.length === 0) {
  console.log("no golden files found; nothing to check");
  process.exit(0);
}
let allOk = true;
for (const f of files) allOk &&= checkMuzero(f);
console.log(allOk ? "golden checks: ALL PASS" : "golden checks: FAILURES");
process.exit(allOk ? 0 : 1);
