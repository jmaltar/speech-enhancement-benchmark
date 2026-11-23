# The predictions and their outcomes

For each of five enhancers on each of four boards, the cost model (`budget_model.py`) gave a
verdict, *fits* or *does not fit*, and a predicted real-time factor (RTF). The predictions were
recorded by 12 September 2026, before any enhancer was ported. *Fits* means that the enhancer
meets the deadline of one hop and that the board has the flash and the RAM for it.

This repository was created when the paper was prepared for submission. It has no version history from the
time of the predictions, so the dates below are those of the project's own records.

## Classical enhancers and RNNoise

Predicted verdict and RTF, then the measured RTF (the median over the 824 test utterances of
VoiceBank-DEMAND, from `analysis/results/cost.csv`) and the date of the measurement.

| board | enhancer | predicted | measured | date | outcome | measured / predicted |
|---|---|---|---|---|---|---|
| F411 | spectral subtraction | fits, 0.064 | 0.111 | 2026-10-14 | held | 1.7 |
| F411 | Wiener filter | fits, 0.064 | 0.112 | 2026-10-14 | held | 1.8 |
| F411 | MMSE-STSA | fits, 0.117 | 0.189 | 2026-10-14 | held | 1.6 |
| F411 | RNNoise | does not fit, 1.116 | 3.17 | 2026-10-26 | held | 2.8 |
| H723 | spectral subtraction | fits, 0.007 | 0.0144 | 2026-10-14 | held | 2.1 |
| H723 | Wiener filter | fits, 0.006 | 0.0145 | 2026-10-14 | held | 2.2 |
| H723 | MMSE-STSA | fits, 0.013 | 0.0252 | 2026-10-14 | held | 2.0 |
| H723 | RNNoise | fits, 0.113 | 0.323 | 2026-10-27 | held | 2.9 |
| S3 | spectral subtraction | fits, 0.031 | 0.085 | 2026-10-14 | held | 2.7 |
| S3 | Wiener filter | fits, 0.027 | 0.085 | 2026-10-14 | held | 3.1 |
| S3 | MMSE-STSA | fits, 0.062 | 0.139 | 2026-10-14 | held | 2.2 |
| S3 | RNNoise | fits, 0.206 | 1.31 | 2026-10-28 | **falsified** | 6.4 |
| Pi 5 | spectral subtraction | fits, 0.001 | 0.0019 | 2026-10-16 | held | 3.0 |
| Pi 5 | Wiener filter | fits, 0.001 | 0.0019 | 2026-10-16 | held | 2.9 |
| Pi 5 | MMSE-STSA | fits, 0.001 | 0.0024 | 2026-10-16 | held | 2.8 |
| Pi 5 | RNNoise | fits, 0.009 | 0.0170 | 2026-10-26 | held | 1.9 |

The ratios are computed from the predicted and measured cycle counts, so a ratio can differ
from the quotient of the two rounded numbers beside it.

RNNoise is the released code without changes. With its weights copied from the flash to the RAM
and the single-precision math library, RNNoise meets the deadline on the S3 (RTF 0.898).

## DTLN

| board | predicted | outcome | date |
|---|---|---|---|
| F411 | does not fit (processing time and flash) | **held.** Model and inference library need 1,162,548 bytes and 476,088 are free. Not run. | 2026-09-16 |
| H723 | fits, RTF 0.149 | **falsified.** The released model (1,027,232 bytes) is larger than the free flash (996,320 bytes), before the inference library (135,316 bytes) is counted. A pruned model fits and runs at RTF 0.389. | 2026-09-16 (flash), 2026-10-21 (pruned model) |
| S3 | does not fit (flash) | **held.** The released model does not fit the default partition of the flash. The pruned model fits it and misses the deadline: RTF 4.59 with the reference kernels of TensorFlow Lite Micro, 1.95 with a 16-bit kernel that we wrote. | 2026-09-16 (flash), 2026-10-22 and 2026-10-26 (pruned model) |
| Pi 5 | fits, RTF 0.003 | **held.** RTF 0.162 for the model at its released size and 0.065 for the pruned model. | 2026-10-27 |

Two notes on the DTLN row.

- The predicted RTF assumed a hop of 16 ms. The model works with a hop of 8 ms, at which the
  prediction for the Pi 5 is 0.0063. The measured cost of the unpruned model is 26 times that.
- The prediction for the H723 was made with the size of the weights alone. The flash that the
  model file and the inference library need was measured four days later, and `budget_model.py`
  was then corrected to count both. The script therefore prints *does not fit* for this
  combination today. The verdict as it was recorded, *fits*, is the one that counts here.

## Summary

Eighteen of the twenty verdicts held and two were falsified. Every measured cost is higher than
the predicted one.
