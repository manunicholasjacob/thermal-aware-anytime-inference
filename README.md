# The Narrow Regime of Thermal-Aware Anytime Inference

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22163329.svg)](https://doi.org/10.5281/zenodo.22163329)

Data and code for an operating-regime study of thermal-aware anytime (early-exit) inference on a
Raspberry Pi 5 (Arm Cortex-A76).

The common claim is that an inference runtime can shed heat by dropping to a shallower exit. It
can, but only inside a narrow regime, and this repository contains the measurements that bound
it. Negative results are included rather than dropped, because the boundary of the regime is the
result.

## Correction, October 2026

**The controller result below was wrong, and the corrected version is the opposite.**
This section is kept at the top rather than folded into the history, because the
original claim was public and people may have read it.

The original campaign ran its policy arms in a fixed order. Re-running the identical
experiment with the arm order randomised moved the controller from 33.49% accuracy to
26.13%, while every other policy reproduced to the hundredth. Only the stateful policy
moved. A third campaign, driving each run to a controlled starting temperature,
explains why.

**What is true:**

- The adaptive controller is **not** the only policy that does both things, and it is
  **dominated** by a standard confidence-threshold exit policy of the kind BranchyNet
  introduced. At an entropy threshold of 3.0 that baseline reaches 29.09% accuracy at
  47.42 C against the controller's 26.13% at 47.77 C: more accurate *and* cooler.
- **The thermal cap acts as a latch.** The controller held full final-exit accuracy in
  14 of 15 runs that began below the 49 C cap, and in 0 of 5 that began at or above it.
- **The accuracy cliff is at the first exit, not the second.** Moving from 100% to 75%
  of requests on the final exit costs 0.23 accuracy points, because the second exit is
  worth 32.75 against the final exit's 33.65. Accuracy only collapses when the
  controller reaches the first exit, worth 19.45, which happens only from a start at or
  above the cap.
- **A single run cannot characterise this controller**, and neither can a campaign with
  a fixed arm order. Across the original run and the two later campaigns the same
  controller, on the same board under the same cap and request rate, measured 23.9%,
  33.49% and 26.13%.

The original claim is left in place below, marked, so the correction can be checked
against what it corrects. Data for all three campaigns is in `data/`, and the harnesses
and analyses are in `code/`.

## What the measurements say

**Under saturating load, exit depth does not control temperature.** Every exit pegs the CPU and
converges to roughly 62 degrees C. The thermal lever exists only at fixed sub-saturation request
rates. A controller that assumes otherwise is steering something it does not have hold of.

**With a non-binding cap (56 C), there is nothing to trade.** The board runs cool, so the
controller converges to the deepest exit and behaves exactly like the static policy it replaced.

**With a binding cap (49 C), the adaptive controller stays thermally safe and beats the safe
fallback on accuracy** (originally measured at 2 s of cap violations against 80 s for
always-final, and 23.9% against 19.5%). ~~That is the regime where anytime inference pays.~~
**SUPERSEDED, see the Correction above.** The original sentence claimed this was the only policy
that does both things. It is not: a confidence-threshold exit policy does both better, and the
controller's own numbers depend on the order the experiment's arms were run in.

**Adaptivity is not free.** The controller's p99/p50 latency ratio is 5.4x, against 1.09 to 1.24x
for every static policy. Its median sits on the shallowest exit (0.1225 s against exit 1's
0.1210 s) while its p99 climbs to 0.660 s, between always-final's median and its p99. A latency
SLO written against the median will be missed by the tail.

**INT8 has its own narrow regime, and it interacts with the first one.** Exported in ONNX
QOperator format the quantized exits are slower than FP32; QDQ export with correct calibration
recovers a 2.46x speedup at the final exit, but the benefit only appears from the second exit
onward, so quantizing the shallowest head is actively negative. The consequence matters for the
thermal work: quantization compresses the exit ladder's latency span from 20.6x to 6.9x,
shrinking the very range the controller trades against the cap. Calibration moves accuracy and
not latency (0.5847 to 0.5965 s across every config, a 2% spread), and more calibration data is
worse, with MinMax-256 beating MinMax-1024 by 0.69 points and MinMax-2048 by 1.08.

Per-exit INT8 ratios are reported in the manuscript rather than restated here, because this
repository ships the calibration sweep and the controller runs, not the per-exit A/B records.

## Layout

```
data/
  onnx_models.tgz               the exported exit heads, FP32 and both INT8 formats
  paper3_clean_dataset.json     the evaluation subset, after cleaning
  paper3_controller_results.json  controller runs at both thermal caps
  paper3_qdq_calib.json         calibration sweep, MinMax at 256/1024/2048
  paper3_controller_reps.json   five repetitions of each controller arm
  paper3_cap_sweep.json         the thermal-cap sweep, with its run log
  paper3_band_sweep.json        the band-width sweep, with its run log
  paper3_entropy_calib.json     exact accuracy and exit mix at every entropy threshold
  paper3_entropy_sweep.json     campaign 1, nine arms, fixed order (the flawed one)
  paper3_entropy_sweep2.json    campaign 2, nine arms, randomised order
  paper3_ratchet_sweep.json     campaign 3, starting temperature controlled
code/
  paper3_thermal_probe.py       thermal characterisation under load
  paper3_controller_exp.py      the adaptive controller experiment
  paper3_int8_ab.py             QOperator against QDQ, per exit
  paper3_qdq_calib.py           calibration sweep
  paper3_qdq_reexport.py        re-export in QDQ format
  paper3_clean_dataset.py       dataset cleaning
  paper3_final_accuracy.py      accuracy at each exit
  paper3_preproc_sweep.py       preprocessing sensitivity
  paper3_reps.py                the repetition campaign
  paper3_cap_sweep.py           the thermal-cap sweep
  paper3_band_sweep.py          the band-width sweep
  fill_reps.py, fill_capsweep.py, fill_bandsweep.py
                                regenerate the manuscript's macro files
  paper3_entropy_calib.py       logits for all exits, exact threshold curves
  paper3_entropy_sweep.py       campaign 1 and the entropy-cascade policy
  paper3_entropy_sweep2.py      campaign 2, randomised and position-balanced order
  paper3_ratchet_sweep.py       campaign 3, starting temperature driven to targets
  paper3_entropy_analyze.py     the dominance analysis
  paper3_ratchet_analyze.py     the two-slope test, written before the data existed
  gen_fig_p3.py, gen_fig_p3b.py, gen_fig_p3c.py
                                figures
paper/
  numbers_reps.tex, numbers_capsweep.tex, numbers_bandsweep.tex
                                generated, committed as generated
```

## Reproducing

Python 3.11 or newer, standard library only, plus `matplotlib` for the figures.

```bash
python code/fill_reps.py        # controller repetitions
python code/fill_capsweep.py    # the thermal-cap sweep
python code/fill_bandsweep.py   # the band-width sweep
python code/gen_fig_p3b.py      # figure
```

The three `paper/numbers_*.tex` files these write are the ones the manuscript includes, and they
are committed here as generated. Running the commands above and then `git diff` is therefore the
check: it comes back empty.

The band sweep is the one worth reading closely, because it is a negative result that the paper
keeps rather than drops. Within-configuration spread is 0.74 of the across-configuration spread,
so the band width is not separable from run-to-run variation on this board.

## Hardware and provenance

All measurements are from one Raspberry Pi 5 (Arm Cortex-A76, 2 GB), running ONNX Runtime on the
CPU execution provider. Temperatures come from the board's own thermal zone. Energy is not
measured here; the PMIC work lives in the sibling energy repositories.

Because every number comes from a single board, treat absolute temperatures as specific to this
unit and its cooling, and the orderings and ratios as the transferable part.

## Citation

See `CITATION.cff`, or cite the concept DOI
[10.5281/zenodo.22163329](https://doi.org/10.5281/zenodo.22163329), which always resolves to the latest version.

## License

MIT, see `LICENSE`.
