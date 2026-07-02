# nano_muzero

Delete `game.apply(s, a)` from the search; learn the simulator instead. Single-file pieces:
`baseline.py` (the AlphaZero capstone, re-run), `model.py` (h/g/f), `mcts.py` (latent search),
`train.py` (unrolled loss, self-play, Reanalyze). Everything trains on a Mac CPU in minutes.

## The data comes first

Before any model: this is what MuZero will ever get to see. One stored self-play game from the
frozen replay (`python -m nano_muzero.show_replay --game 3`), exactly as the trainer reads it:

```
replay: data/replays/ttt_capstone.npz
  schema ttt-replay-v1, 500 games, 3023 positions, keys: obs board to_play action pi z game_id move_idx
  z labels (from each mover's view): win 1344, draw 738, loss 941
  mean pi entropy 1.03 nats (uniform would be 2.20)

game 3: 7 moves
   ...   ...   .O.   .O.   .O.   .O.   .OO
   ...   .X.   .X.   XX.   XX.   XX.   XX.
   ...   ...   ...   ...   .O.   XO.   XO.
   X:4   O:1   X:3   O:7   X:6   O:2   X:0

   move  mover  action  z      pi over cells 0..8
      0      X       4  +1    |    ▇    |  max pi(4)=0.84
      1      O       1  -1    |      ▃ ▁|  max pi(6)=0.44
      2      X       3  +1    |  ▁▃  ▁ ▁|  max pi(3)=0.34
      3      O       7  -1    |      ▂▃▁|  max pi(7)=0.34
      4      X       6  +1    |▁     ▅ ▂|  max pi(6)=0.56
      5      O       2  -1    |▂ ▃  ▁  ▁|  max pi(2)=0.36
      6      X       0  +1    |▅    ▂   |  max pi(0)=0.64
```

Read it like the model does. Four things are stored per position, and they are the whole diet:

- **`obs`** (rendered above as boards): the capstone's 2-plane encoding, always from the
  to-play player's view. There is no other observation; MuZero never sees "the rules".
- **`pi`** (the bar strips): the MCTS visit distribution when that move was chosen -- the
  *search-improved* policy, not the move alone. Sharp bars (`pi(4)=0.84`) are positions where
  the search was sure; flat bars are exploration noise doing its job.
- **`z`** (the +1/-1 column): the final game outcome re-signed to each position's mover.
  X won this game, so every X row carries +1 and every O row -1. That is the value target.
- **`action`**: what was actually played (the teacher-forcing rail for the K-step unroll).

The mix matters: 500 noisy self-play games give 1344/738/941 win/draw/loss labels and plenty
of imperfect lines (watch O never block the column in game 3). Diversity is what the learned
dynamics `g` trains on; a perfect-play-only replay would teach it almost nothing off the
main line.

Regenerate everything (both the replay and the checkpoint it came from, ~40 s on CPU):

```
python -m nano_muzero.baseline                                # train + write replay
python -m nano_muzero.baseline --games 200 --assert-never-loses   # the M1.a gate
python -m nano_muzero.show_replay --game 3                    # look at the data
```

## The baseline it must beat (M1.a)

`baseline.py` is the AlphaZero capstone (courses/alphazero/capstone/AGENT.md) re-run with the
identical Game interface from `envs/tictactoe.py`. Gate, asserted not eyeballed: over 200
arena games vs a random mover at 200 simulations it never loses (last run: 192 wins, 8 draws,
0 losses). nano_muzero's exit exam is arena parity against this exact agent.

Value convention (D1, recorded in DECISIONS.md): value head is `tanh` in [-1, 1], targets
z in {-1, 0, +1} from the mover's view, MSE loss. The AlphaZero deck's [0,1]+BCE presentation
is a deck convention and appears there as a labeled aside.

## Milestone ladder

| milestone | what runs | gate | status |
|---|---|---|---|
| M1.a | `baseline.py`: capstone re-run + frozen replay | never-loses; replay schema test | PASS (192W/8D/0L) |
| M1.b | `model.py`: h/g/f | shape + NaN-free unroll tests | PASS |
| M1.c | `mcts.py`: latent MCTS + oracle harness | move-for-move match with capstone MCTS | PASS, stronger than promised: EXACT root visit-count vectors, 20/20 |
| M1.d | `train.py --offline` on the frozen replay | losses fall; probe + drift curves | PASS |
| M1.e | full self-play loop (`--full`) | arena parity with M1.a | 4/5 gates PASS (below) |
| M1.f | `--reanalyze` under constrained data | seeded comparison table | measured, sign FLIPPED (experiments.md) |
| M1.g | break-it suite, `--noisy` env | experiments table | measured (experiments.md) |

The shipped checkpoint (`data/ckpts/muzero_selfplay.pt`, recipe in its run manifest:
warm-start from M1.d + mild noise 0.1 + `--mix-replay` + `--vs-random-frac 0.3`) gates at
200 sims as: **arena vs capstone 33W 57D 10L, net +23 against a mirror baseline of +4**
(superiority, not just parity; at 50 sims it concedes ZERO losses in 100 games), empty-board
root value +0.087, oracle harness 20/20, search beats its own raw policy vs the capstone
(+33 vs +17). The one red gate: 18/200 losses to a uniform-random mover -- 17 of them as O,
all one-ply blocking misses in off-distribution corner openings; scaling vs-random data 30%
-> 50% did not shrink the tail (19/200). Model-blind-spot losses to weak opponents are, so
far, the honest price of planning inside a learned simulator at this scale; the 1.8 lab's
hallucination finder exists to let you inspect exactly these.

Getting here required un-learning a delusion, and the story ships with the module: pure
noisy self-play converges to "X always wins" (38/40 X wins in its own games, empty-board
value +0.6) because -- unlike your capstone, whose true-rules search injects tactical truth
no matter how bad the net is -- MuZero's search can only be as honest as its model. Fully
cooled self-play calibrates perfectly (+0.001) and then loses to random 72/200 from pure
coverage starvation. The working recipe anchors on the frozen M1.a replay (MuZero
Unplugged's offline lesson) and seeds off-policy coverage games. Sims-budget crossover,
measured: the offline model BEATS the capstone at matched 50-sim search (+36 net) and loses
at 200 (-29) -- the paper's Fig. 3 Atari plateau, reproduced on a 3x3 board.

Measured highlights (all reproducible with the commands above, seeds printed):

- **Oracle harness (M1.c):** with min-max normalization off and the capstone's constants,
  the latent search over true dynamics produces bit-identical root visit vectors to the
  capstone MCTS on all 20 fixed positions (`python -m nano_muzero.oracle`). With MuZero's
  own c1/c2 + min-max normalization the same positions agree on 17/20 moves at 200 sims:
  the normalizer legitimately trades low-sim tactical sharpness for reward-scale freedom.
- **Unroll drift (M1.d, the argument for K-step training):** teacher-forced value error at
  depth 6-7 is 0.56-0.59 for the K=1-trained model vs 0.11-0.27 for K=5
  (`python -m nano_muzero.eval_drift`, exported to `data/eval/drift_curves.json`).
- **Value equivalence, measured (M1.d):** a linear probe decodes board occupancy from
  frozen latents at 0.930 accuracy (by accident) while the value head reads the same
  latents at 0.991 sign accuracy (by construction). Adding a reconstruction loss
  (`--recon on`) pushes occupancy to 0.978 and buys zero value accuracy. Killing the
  value+reward losses (`--ablate-value`) collapses value sign accuracy to 0.568 while
  occupancy persists at 0.927 (`python -m nano_muzero.probe`, experiments.md).

*90% of nano_muzero bugs are target misalignment: an off-by-one between pi_{t+k} / z_{t+k} /
u_{t+k} and the k-th unroll head. The oracle harness (M1.c) is how you notice.*
