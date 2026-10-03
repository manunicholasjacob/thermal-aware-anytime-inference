#!/usr/bin/env python3
"""Paper 3: calibrate the entropy-threshold exit policy before measuring it.

Why this exists. The paper compares its thermal controller against always-exit1,
always-exit2 and always-final, which are its own static bounds rather than
published work. The baseline the early-exit literature actually uses, from
BranchyNet onward, is a confidence threshold on the side-branch output: take the
shallow answer when the branch is confident, escalate when it is not. TETC asks
for a comparison against state-of-the-art work and that is the comparison.

This script does the cheap half. It runs all three exits over the whole CIFAR-100
test set once and stores the logits, so that the accuracy and the exit mix at any
threshold are then exact and free to compute. Choosing the threshold grid from
the measured entropy distribution beats guessing it, and the expensive thermal
runs then only have to measure temperature and latency.

Two things it checks rather than assumes:

  1. Whether the ONNX graphs emit logits or probabilities. Applying softmax to
     something already normalised would silently distort every entropy.
  2. That the per-exit accuracies it computes agree with the ACC table hard-coded
     in paper3_controller_exp.py. If they disagree, the stored artifacts are not
     the ones the paper's numbers came from and nothing downstream is valid.

Light job: one pass over 10k images at about 0.85 ms total per image, so under a
minute, with no sustained thermal load.
"""
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

M = Path.home() / "tier1-experiments/models"
DATA = Path.home() / "Desktop/researchpaper3/data"
OUT_NPZ = Path.home() / "paper3_entropy_logits.npz"
OUT_JSON = Path.home() / "paper3_entropy_calib.json"

EXITS = ["exit1", "exit2", "final"]
# measured in the paper, used here only as a cross-check
ACC_PAPER = {"exit1": 19.45, "exit2": 32.73, "final": 33.65}

with tarfile.open(DATA / "cifar-100-python.tar.gz", "r:gz") as t:
    d = pickle.loads(t.extractfile("cifar-100-python/test").read(), encoding="latin1")
RAW = np.asarray(d["data"], dtype=np.uint8)
Y = np.array(d["fine_labels"], dtype=np.int64)
N = len(Y)


def img(i):
    return RAW[i:i + 1].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0


def mksess(ex):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(M / ("mnv3_c100_%s_qdq_clean.onnx" % ex)),
                                sess_options=so, providers=["CPUExecutionProvider"])


SESS = {ex: mksess(ex) for ex in EXITS}
INAME = SESS["final"].get_inputs()[0].name

print("collecting logits for %d images at %d exits" % (N, len(EXITS)), flush=True)
logits = {ex: np.zeros((N, 100), dtype=np.float32) for ex in EXITS}
for i in range(N):
    x = img(i)
    for ex in EXITS:
        logits[ex][i] = SESS[ex].run(None, {INAME: x})[0][0]
    if (i + 1) % 2000 == 0:
        print("  %d/%d" % (i + 1, N), flush=True)

# --- check 1: logits or probabilities? ---
rowsum = float(np.abs(logits["final"][:50].sum(axis=1) - 1.0).mean())
already_prob = rowsum < 1e-3
print("\nmean |rowsum-1| on final, first 50 rows: %.6f -> %s"
      % (rowsum, "ALREADY PROBABILITIES" if already_prob else "logits, softmax needed"),
      flush=True)


def probs(z):
    if already_prob:
        return np.clip(z, 1e-12, 1.0)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return np.clip(e / e.sum(axis=1, keepdims=True), 1e-12, 1.0)


P = {ex: probs(logits[ex]) for ex in EXITS}
H = {ex: -(P[ex] * np.log(P[ex])).sum(axis=1) for ex in EXITS}  # nats
PRED = {ex: P[ex].argmax(axis=1) for ex in EXITS}

# --- check 2: do the per-exit accuracies match the paper's table? ---
print("\nper-exit accuracy, measured here vs the paper's hard-coded table:", flush=True)
acc_now = {}
ok = True
for ex in EXITS:
    a = 100.0 * float((PRED[ex] == Y).mean())
    acc_now[ex] = a
    delta = a - ACC_PAPER[ex]
    flag = "ok" if abs(delta) < 0.75 else "MISMATCH"
    if flag == "MISMATCH":
        ok = False
    print("  %-6s measured %6.2f%%   paper %6.2f%%   delta %+5.2f  %s"
          % (ex, a, ACC_PAPER[ex], delta, flag), flush=True)

print("\nentropy distribution, nats, max possible ln(100)=%.3f" % np.log(100), flush=True)
for ex in ["exit1", "exit2"]:
    q = np.percentile(H[ex], [1, 5, 10, 25, 50, 75, 90, 95, 99])
    print("  %-6s p1=%.3f p5=%.3f p10=%.3f p25=%.3f p50=%.3f p75=%.3f p90=%.3f p95=%.3f p99=%.3f"
          % tuple([ex] + list(q)), flush=True)


def cascade(tau):
    """Exact accuracy and exit mix for threshold tau, over the whole test set."""
    take1 = H["exit1"] <= tau
    take2 = (~take1) & (H["exit2"] <= tau)
    take3 = ~(take1 | take2)
    pred = np.where(take1, PRED["exit1"], np.where(take2, PRED["exit2"], PRED["final"]))
    return dict(tau=round(float(tau), 4),
                acc=100.0 * float((pred == Y).mean()),
                f1=100.0 * float(take1.mean()),
                f2=100.0 * float(take2.mean()),
                f3=100.0 * float(take3.mean()))


print("\naccuracy and exit mix against threshold, exact over the full test set:", flush=True)
print("  %6s %7s %7s %7s %7s" % ("tau", "acc%", "exit1%", "exit2%", "final%"), flush=True)
grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5,
        2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.605]
curve = [cascade(t) for t in grid]
for c in curve:
    print("  %6.3f %7.2f %7.1f %7.1f %7.1f"
          % (c["tau"], c["acc"], c["f1"], c["f2"], c["f3"]), flush=True)

np.savez_compressed(OUT_NPZ, **{("H_" + k): v for k, v in H.items()},
                    **{("pred_" + k): v for k, v in PRED.items()}, y=Y)
json.dump(dict(n=int(N), already_prob=bool(already_prob), acc_measured=acc_now,
               acc_paper=ACC_PAPER, acc_table_agrees=bool(ok), curve=curve),
          open(OUT_JSON, "w"), indent=1)
print("\nwrote %s and %s" % (OUT_NPZ, OUT_JSON), flush=True)
if not ok:
    print("\nSTOP: per-exit accuracy does not match the paper. The stored models are "
          "not the ones the paper's numbers came from. Do not run the thermal sweep.",
          flush=True)
