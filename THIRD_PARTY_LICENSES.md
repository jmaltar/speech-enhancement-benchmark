# Third-party components

The third-party code in this repository is kept exactly as it was built and measured, so that
the firmware images can be rebuilt. Each component keeps its own licence file in place.

| component | version | location | licence | licence file |
|---|---|---|---|---|
| Arm CMSIS (Core, DSP 1.10.0) | from STM32Cube FW_F4 V1.28.3 and FW_H7 V1.13.0 | `benchmark_{f411,h723}/Drivers/CMSIS/`, `Middlewares/Third_Party/CMSIS-DSP/` | Apache-2.0 | `Drivers/CMSIS/LICENSE.txt` |
| STM32F4xx CMSIS device headers | FW_F4 V1.28.3 | `benchmark_f411/Drivers/CMSIS/Device/ST/STM32F4xx/` | Apache-2.0 | `LICENSE.md` there |
| STM32H7xx CMSIS device headers | FW_H7 V1.13.0 | `benchmark_h723/Drivers/CMSIS/Device/ST/STM32H7xx/` | Apache-2.0 unless a package licence applies, see the file | `LICENSE.txt` there |
| STM32F4xx and STM32H7xx HAL drivers | FW_F4 V1.28.3, FW_H7 V1.13.0 | `benchmark_{f411,h723}/Drivers/STM32*_HAL_Driver/` | BSD-3-Clause (STMicroelectronics) | `LICENSE.md` / `LICENSE.txt` there |
| TinyUSB | 0.21.0 | `benchmark_{f411,h723}/Middlewares/Third_Party/tinyusb/`, `benchmark_s3/managed_components/espressif__tinyusb/` | MIT | `LICENSE` there |
| esp-dsp | 1.5.2 | `benchmark_s3/managed_components/espressif__esp-dsp/` | Apache-2.0 | `LICENSE` there |
| esp-nn | 1.3.2 | `benchmark_s3/managed_components/espressif__esp-nn/` | Apache-2.0 | `LICENSE` there |
| esp_tinyusb | 1.7.6 | `benchmark_s3/managed_components/espressif__esp_tinyusb/` | Apache-2.0 | `LICENSE` there |
| RNNoise (Xiph.Org, Jean-Marc Valin) | v0.1.1 and v0.1 | `rnnoise/xiph-v0.1.1/`, `rnnoise/xiph-v0.1/` | BSD-3-Clause | `COPYING` there |
| DTLN pretrained models and model code (Nils L. Westhausen) | `github.com/breizhn/DTLN` | `dtln/pretrained/`, `dtln/upstream/` | MIT | `dtln/pretrained/LICENSE.DTLN`, `dtln/upstream/LICENSE` |

Not included, obtained from upstream when needed:

| component | version | licence |
|---|---|---|
| TensorFlow Lite Micro, with CMSIS-NN V.6.2.0 | commit `64b3de8` of `tensorflow/tflite-micro` (and `d0dabe5` for the flash measurement in `tflm_footprint/`) | Apache-2.0 |
| Open-Unmix, `umxhq` weights | openunmix 1.3.0, downloaded when `separation/umx_run.py` runs | MIT |
| ITU-T P.862 reference code | through the `pesq` 0.0.4 Python package | see the package |

The files that STM32CubeMX generated (`Core/`, the `.ioc` files) are under STMicroelectronics'
terms for generated code, as stated in their headers. The models in `dtln/` and `training/`
that we exported are derived from the DTLN pretrained models (MIT).

The corpora are not redistributed. Their sources and licences are listed in `README.md`.
