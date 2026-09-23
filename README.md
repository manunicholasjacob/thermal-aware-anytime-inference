# The Narrow Regime of Thermal-Aware Anytime Inference

Data and code for an operating-regime study of thermal-aware anytime (early-exit) inference on a
Raspberry Pi 5 (Arm Cortex-A76).

The common claim is that an inference runtime can shed heat by dropping to a shallower exit. It
can, but only inside a narrow regime, and this repository contains the measurements that bound
it. Negative results are included rather than dropped, because the boundary of the regime is the
result.

## What the measurements say

**Under saturating load, exit depth does not control temperature.** Every exit pegs the CPU and
converges to roughly 62 degrees C. The thermal lever exists only at fixed sub-saturation request
rates. A controller that assumes otherwise is steering something it does not have hold of.

**With a non-binding cap (56 C), there is nothing to trade.** The board runs cool, so the
controller converges to the deepest exit and behaves exactly like the static policy it replaced.

**With a binding cap (49 C), the adaptive controller is the only policy that does both things.**
It stays thermally safe, 2 s of cap violations against 80 s for always-final, and it beats the
safe fallback on accuracy, 23.9% against 19.5%. That is the regime where anytime inference pays.

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
  gen_fig_p3.py, gen_fig_p3b.py figures
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

See `CITATION.cff`, or use the DOI badge once the first release is archived.

## License

MIT, see `LICENSE`.
