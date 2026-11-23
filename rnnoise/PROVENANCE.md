# RNNoise — provenance of the vendored sources under `rnnoise/`, recorded 2026-10-19

**Source:** `github.com/xiph/rnnoise`, cloned 2026-10-19 (tip `70f1d25`, 2025-02-22). Two release
tags are vendored with `git archive <tag> src include COPYING README` (training script, build
helper and nothing else removed); BSD-3-Clause, `COPYING` kept in each tree.

| tree | tag | commit | date | role |
|---|---|---|---|---|
| `xiph-v0.1.1/` | `v0.1.1` | `babe7fb` | 2024-03-22 | **the as-shipped release** — what a practitioner links today; the oracle's primary variant (`rnnoise`) |
| `xiph-v0.1/` | `v0.1` | `2cb7786` | 2017-09-19 | **the paper-era tree** — the day of the last model retrain the paper describes; variant `rnnoise-v01` |

**The model is identical in both:** `src/rnn_data.c` differs only by `static` qualifiers and the
`rnnoise_model_orig` struct added by the 2018 modularisation; **15 weight arrays, 87,503 `signed char`
weights at `WEIGHTS_SCALE = 1/256`** — the figure the budget was costed against and the paper states.
`sha256(xiph-v0.1.1/src/rnn_data.c) = 7b90ed18f065a34597d80b4c00e140a7c5eb6380ef8c9c6bf265384c944ae708`.

| array | weights |
|---|---:|
| input_dense (weights / bias) | 1,008 / 24 |
| vad_gru (input / recurrent / bias) | 1,728 / 1,728 / 72 |
| noise_gru | 12,960 / 6,912 / 144 |
| denoise_gru | 32,832 / 27,648 / 288 |
| denoise_output | 2,112 / 22 |
| vad_output | 24 / 1 |

**What v0.1.1 changes besides the model:** the **gain-decay limiter** of commit `91ef401` (2017-10-10,
"Limiting the gain decay"): `g[i] = max(g[i], 0.6 · lastg[i])` per band after the pitch filter — an
algorithmic change made three weeks after v0.1 and present in every release since, but **not in the
tree the paper's experiments ran on**; plus the `RNNModel` API (`rnnoise_create(NULL)` selects the
built-in model), `rnnoise_get_frame_size()`, `rnn_`-prefixed internal symbols and maintenance fixes.
Its effect is measured in `RESULTS.md`: 22–40 % of output samples differ by up to 1,560 LSB, and
every metric is TOST-equivalent.

**Not vendored:** `v0.2` (2024-04-14) and later, which ship a different, larger retrained model.
Nothing in this project was costed against it.

**Build:** `cmake -S rnnoise -B rnnoise/build && cmake --build rnnoise/build` — Release `-O2 -DNDEBUG`
set after `project()`, `-ffast-math` declined, vendored code compiled as shipped (no warning flags of
ours), `HAVE_CONFIG_H` undefined so no `config.h` is consulted, `TRAINING` 0. Produces `rnnoise_wav`
(v0.1.1) and `rnnoise_wav_v01`.
