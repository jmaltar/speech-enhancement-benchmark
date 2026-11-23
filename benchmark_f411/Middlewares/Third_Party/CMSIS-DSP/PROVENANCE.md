# CMSIS-DSP — vendored subset

**Version 1.10.0**, taken from STM32Cube FW_F4 **V1.28.3**
(`Drivers/CMSIS/DSP` of `~/STM32Cube/Repository/STM32Cube_FW_F4_V1.28.3`).

Vendored rather than referenced out of the Cube repository so the build is
reproducible from this tree alone (task 26), and so the library version is
pinned alongside the CubeMX / FW-pack versions the memory figures depend on.

Only the translation units the throughput probe needs are copied. The whole
`Source/` tree is NOT copied, deliberately: CMSIS-DSP 1.10 ships both
per-function files (`arm_dot_prod_f32.c`) and per-folder aggregates that
`#include` them all (`BasicMathFunctions.c`). Globbing a directory links both
and fails on duplicate symbols.

`Include/` and `PrivateInclude/` are copied whole.

## Files and why each is here

| File | Needed by |
|---|---|
| `TransformFunctions/arm_cfft_f32.c` | `cmsis_cfft_{256,512}` — the like-for-like pair for the plain-C complex FFT |
| `TransformFunctions/arm_cfft_init_f32.c`, `arm_cfft_radix8_f32.c`, `arm_bitreversal2.c` | pulled in by `arm_cfft_f32` |
| `TransformFunctions/arm_rfft_fast_f32.c`, `arm_rfft_fast_init_f32.c` | `cmsis_rfft_512` — the real transform the STFT path actually uses |
| `CommonTables/arm_common_tables.c` | twiddle / bit-reversal tables (4.7 MB of source; `--gc-sections` keeps only the sizes used) |
| `CommonTables/arm_const_structs.c` | `arm_cfft_sR_f32_len256/512` instances |
| `BasicMathFunctions/arm_dot_prod_f32.c` | pair for `dot_f32_512` |
| `BasicMathFunctions/arm_mult_f32.c` | pair for `window_f32_512` |
| `ComplexMathFunctions/arm_cmplx_mag_f32.c` | pair for `mag_f32_257` |
| `MatrixFunctions/arm_mat_vec_mult_f32.c` | pair for `matvec_f32_{64,128}` |
| `MatrixFunctions/arm_mat_vec_mult_q7.c` | pair for `matvec_s8_{128,256}` — the SMLAD/SXTB16 kernel the compression cost model rests on |
| `FastMathFunctions/arm_vexp_f32.c`, `arm_vlog_f32.c` | pairs for `expf_257` / `logf_257` |

`arm_vsqrt_f32` does not exist in 1.10.0, so `sqrtf_257` has no CMSIS pair;
`arm_sqrt_f32` is a scalar inline over the FPU's `VSQRT.F32`, which is what
libm's `sqrtf` already compiles to.
