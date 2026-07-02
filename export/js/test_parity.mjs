/* Search parity: the JS searches (search.js) must pick the same moves as the Python ones
 * (mcts.py / baseline.py) on the 20 oracle positions, using the exported weights.
 * Forward-pass parity is already golden-checked to 1e-5; this checks the whole search.
 * PyTorch computes in float32 and JS in float64, so visit COUNTS may differ on knife-edge
 * ties; the assertion is same argmax everywhere + counts reported.
 *
 *   node export/js/test_parity.mjs
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { AlphaZeroJS } from "./forward.js";
import { MuZeroCfg, MuZeroModelJS, runCapstoneMCTS, runMCTS, TTT } from "./search.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(readFileSync(join(HERE, "..", "golden", "search_parity.json")));
const nanoW = JSON.parse(readFileSync(join(HERE, "..", "golden", ref.weights)));
const capW = JSON.parse(readFileSync(join(HERE, "..", "golden", ref.capstone_weights)));

const cap = new AlphaZeroJS(capW);
const nano = new MuZeroModelJS(nanoW);
const cfg = MuZeroCfg({ nSims: ref.sims });

const argmax = (a) => a.indexOf(Math.max(...a));
let ok = 0, capExact = 0, latExact = 0;
for (const row of ref.positions) {
  const board = row.board;
  const capJS = runCapstoneMCTS(cap, board, ref.sims).counts;
  const latJS = runMCTS(nano, board, TTT.legal(board), cfg).counts;
  const capMatch = argmax(capJS) === argmax(row.capstone_counts);
  const latMatch = argmax(latJS) === argmax(row.latent_counts);
  capExact += JSON.stringify(capJS) === JSON.stringify(row.capstone_counts.map(Number)) ? 1 : 0;
  latExact += JSON.stringify(latJS) === JSON.stringify(row.latent_counts.map(Number)) ? 1 : 0;
  if (capMatch && latMatch) ok++;
  else console.log(`MISMATCH at ${board.join("")}: capstone JS ${argmax(capJS)} vs py ${argmax(row.capstone_counts)}, latent JS ${argmax(latJS)} vs py ${argmax(row.latent_counts)}`);
}
console.log(`search parity: ${ok}/${ref.positions.length} positions same argmax (both searches)`);
console.log(`  exact visit-vector matches: capstone ${capExact}/20, latent ${latExact}/20 (float32 vs float64; exactness not required)`);
process.exit(ok === ref.positions.length ? 0 : 1);
