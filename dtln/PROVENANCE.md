# DTLN — provenance of the artifacts under `dtln/pretrained/`, recorded 2026-10-19

**Source:** `github.com/breizhn/DTLN`, commit `1de1f15a8b5b7e1c44905618ff2ef70ca8277fbc`
(2022-04-26, the tip of `master` on 2026-10-19), fetched with `git clone --depth 1`. MIT licence,
copied beside the models as `LICENSE.DTLN`. Paper: Westhausen & Meyer, *Dual-Signal Transformation
LSTM Network for Real-Time Noise Suppression*, Interspeech 2020, arXiv:2005.07551v2 (on disk in
`papers/`).

The models read on 2026-09-16 for `tflm_footprint/` lived in a session scratchpad and were gone by
2026-10-19; these are the same files, re-fetched, and the sizes agree with what CLAUDE.md recorded
then (3,975,748 B float pair, 1,027,232 B quantized pair).

| file | bytes | sha256 | what it is |
|---|---:|---|---|
| `model.h5` | 3,989,312 | `b6e9ecb8…6ff0d55d` | Keras weights of the DNS-Challenge submission — the paper's Table 2 row. 986,753 parameters (asserted by `check_oracle.py` after loading) |
| `model_1.tflite` | 1,459,944 | `91281a38…1e7a0d1e`¹ | stage 1 (STFT-magnitude mask), float32, external LSTM state. The authors' export of `model.h5` |
| `model_2.tflite` | 2,515,804 | `7ae37ec8…1520830` | stage 2 (learned basis), float32, external state |
| `model_quant_1.tflite` | 377,776 | `5008c212…acce6a6da` | stage 1, dynamic-range int8 (float32 activations, int8 weight tensors) |
| `model_quant_2.tflite` | 649,456 | `a8a18eeb…1b762b4` | stage 2, same |
| `dtln_saved_model/` | — | `saved_model.pb 2bc1149e…`, `variables.data a43f7e9d…`, `variables.index a306f5d4…` | the stateful SavedModel (frame in, frame out, LSTM state kept inside); the independent runtime `check_oracle.py` compares the TFLite pair against |

¹ Full digests, as printed by `sha256sum` in `dtln/pretrained/`:

```
b6e9ecb8140f51e8ffb373884b62e186dc166e97995fa7eea3fffcc6ef30d55d  model.h5
91281a38e80fe9fd330e28eda7e16fe4e483ee5199a3e687a099939013c25de0  model_1.tflite
7ae37ec802862d8a65b5cdabfbcbbe22caaf7cd39e79adf574d15837d1520830  model_2.tflite
5008c2121dbbf3fedd2a475985dac261a10a1fffc97fbacce6a6da1dc8c7ebf6  model_quant_1.tflite
a8a18eebb642d589ae006c554d2d1c09ac1c300e45caf1f19c766d3391b762b4  model_quant_2.tflite
2bc1149e2c9e5e090de1567d9edec735b8f1726d7c600d1b5765a602847d0408  dtln_saved_model/saved_model.pb
a43f7e9dbece1453320b55ea976660d21448682bd0f1804a21be37239e2bf163  dtln_saved_model/variables/variables.data-00000-of-00001
a306f5d4fc32ef95dfcb137870075afd0cb12e98a1ce8e4452bbbc6559cccc9a  dtln_saved_model/variables/variables.index
```

Added 2026-10-19 for the quantization study: `DTLN_norm_500h.h5` (3,989,312 B,
`378c209ad3f4dedffb185b542169a51472361a23492c13b377feb2485a2a1bb3`) — the authors' weights trained
with log-magnitude input normalisation, same architecture plus one instant layer norm (986,753 + 514
parameters). Not copied: `DTLN_norm_40h.h5`, the three SavedModels of the norm variants and the ONNX
exports.

## Generated here — `dtln/quantize.py`, 2026-10-19

Re-exports of the weights above with the LSTM states as **separate tensors** (h0, c0, h1, c1 per
stage) and, for the 16x8 pair, an instant layer norm whose variance is formed as E[x²] − E[x]²
(`ILN_E2`) so no SQUARED_DIFFERENCE is emitted. Calibrated on the 200-utterance `calib=1` subset of
the VoiceBank **train** split, every 16th hop, 4,610 samples per stage. TensorFlow 2.21.0 converter.

| file | bytes | sha256 (prefix) | what |
|---|---:|---|---|
| `model_int8_{1,2}.tflite` | 404,752 + 692,960 | `325ff69a…` / `da4be74e…` | paper weights, full-integer int8, int8 I/O, split states. **Collapses** |
| `model_int8p_{1,2}.tflite` | 499,352 + 787,184 | `1e530d0f…` / `3216472b…` | same, authors' **packed** state layout; the converter leaves the mask Dense in float. **Collapses**; kept as the recorded failure |
| `model_i16_{1,2}.tflite` | 425,488 + 723,928 | `839e2c98…` / `2901e897…` | paper weights, **16x8** (int16 activations, int8 weights), int16 I/O, split states, `ILN_E2`. **The deployable form** |
| `model_f32n_{1,2}.tflite` | 1,461,912 + 2,512,724 | `d195b562…` / `94a023d2…` | norm weights, float32 (their float baseline here) |
| `model_int8n_{1,2}.tflite` | 408,016 + 692,960 | `a3a428e3…` / `4dd0463f…` | norm weights, full-integer int8. Collapses; stage 1 (LOG) does not invoke on TFLM |
| `model_i16n_{1,2}.tflite` | 429,640 + 723,984 | `106ca3b0…` / `ec086e5d…` | norm weights, 16x8. Collapses; stage 1 does not invoke on TFLM |

Full digests are in `results/`'s logs' companion listing (`sha256sum *.tflite` in `pretrained/`).
The float re-export of the paper weights with split states reproduces the shipped pair to 0 LSB,
and with `ILN_E2` to 1 LSB on 4 of 25,526 samples (quantize logs). Those re-exports were deleted
after the check; the shipped pair is the float oracle.

## Runtime, pinned

| component | version | role |
|---|---|---|
| `ai-edge-litert` | 2.3.0 | the TFLite interpreter the oracle runs (`Interpreter`, XNNPACK delegate on by default, `num_threads=1`) |
| `tflite-micro` | 0.dev20261017155528 | **TFLite Micro on the host** (`tflite_micro.python.tflite_micro.runtime`): the MicroInterpreter with the RecordingMicroAllocator. Added 2026-10-19; `dtln/tflm_check.py` and the `-tflm` oracle variants run on it |
| `tflite` | 2.18.0 | flatbuffer schema bindings, for reading op sets and tensor types out of `.tflite` files |
| `tensorflow` | 2.21.0 (Keras 3.15.1) | `check_oracle.py` only: SavedModel and the authors' Keras `model.h5` path. **Also the dependency task 13 (training) needs**, which is why the full package rather than the interpreter alone was installed |
| Python | 3.11.15 (`.venv`, `uv`) | unchanged |

Both were installed into `.venv` on 2026-10-19; `requirements-eval.txt` was regenerated with
`uv pip freeze` and grew from 34 to 64 lines (66 after `tflite-micro` and `tflite` the same day). **None of the 34 previously frozen pins moved** (the
diff has no removed lines). GPU: the CUDA 12.9 libraries (`tensorflow[and-cuda]`, 12 `nvidia-*` packages) were added the
same evening for task 13; TensorFlow sees the GPU only with `source gpu_env.sh` (project root). Nothing in
this directory needs it, and the quantization runs above were CPU-only.

The authors' `DTLN_model.py` is *not* vendored. `check_oracle.py --dtln-src <clone>` imports it
from a clone for the Keras check and stubs its `wavinfo` import (a training-data helper) instead of
installing that package.
