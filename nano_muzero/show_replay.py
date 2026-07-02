"""Render replay games as text: boards, MCTS pi histograms, z labels (M1.a inspection tool).

This is how you look your training data in the eye before believing any model trained on it:
  python -m nano_muzero.show_replay                # stats + first game
  python -m nano_muzero.show_replay --game 7       # any stored game
"""
import argparse
import json

import numpy as np

from nano_muzero.baseline import REPLAY_PATH, validate_replay

GLYPH = {1: "X", -1: "O", 0: "."}
BARS = " ▁▂▃▄▅▆▇█"


def board_lines(board):
    return ["".join(GLYPH[c] for c in board[r * 3 : r * 3 + 3]) for r in range(3)]


def pi_bar(pi):
    return "".join(BARS[min(8, int(p * 8.999))] for p in pi)


def show_game(data, gid: int):
    idx = np.flatnonzero(data["game_id"] == gid)
    print(f"game {gid}: {len(idx)} moves")
    rows = [data["board"][i] for i in idx]
    for line in range(3):
        print("   " + "   ".join(board_lines(b)[line] for b in rows))
    print("   " + "   ".join(f"{GLYPH[int(data['to_play'][i])]}:{data['action'][i]}" for i in idx))
    print()
    print("   move  mover  action  z      pi over cells 0..8")
    for i in idx:
        pi = data["pi"][i]
        print(
            f"   {data['move_idx'][i]:>4}  {GLYPH[int(data['to_play'][i])]:>5}"
            f"  {data['action'][i]:>6}  {data['z'][i]:>+.0f}    |{pi_bar(pi)}|"
            f"  max pi({np.argmax(pi)})={pi.max():.2f}"
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replay", default=REPLAY_PATH)
    ap.add_argument("--game", type=int, default=0)
    args = ap.parse_args(argv)

    data = validate_replay(args.replay)
    meta = json.loads(str(data["meta"]))
    n, games = len(data["obs"]), data["game_id"].max() + 1
    z = data["z"]
    print(f"replay: {args.replay}")
    print(f"  schema {meta['schema']}, {games} games, {n} positions, keys: obs board to_play action pi z game_id move_idx")
    print(f"  z labels (from each mover's view): win {int((z > 0).sum())}, draw {int((z == 0).sum())}, loss {int((z < 0).sum())}")
    pi = data["pi"]
    entropy = -(pi * np.log(np.where(pi > 0, pi, 1.0))).sum(1).mean()
    print(f"  mean pi entropy {entropy:.2f} nats (uniform would be {np.log(9):.2f})")
    print()
    show_game(data, args.game)


if __name__ == "__main__":
    main()
