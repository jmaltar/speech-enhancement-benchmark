/* rnn_nn8.h -- the "+N" RNNoise variant: the three GRUs on CMSIS-NN.
 *
 * What it is. Xiph's compute_gru() is a scalar float loop over the shipped
 * 8-bit weights, read with a byte stride of 3N -- 17 cycles per MAC on the
 * F411 and half of its hop (benchmark_f411/RESULTS.md, 2026-10-30 stamps).
 * This variant computes the same three gates with CMSIS-NN's
 * arm_nn_vec_mat_mult_t_svdf_s8 (V.6.2.0, as vendored in tflm_tree_64b3de8):
 * int8 x int8 with SMLAD on Cortex-M, int32 accumulate, int16 output.
 *
 * What changes numerically, and only this:
 *   - the input vector, the state and state*r are quantised to int8 per call,
 *     symmetric, scale max|x|/127 (no calibration, nothing learned);
 *   - the kernel stores round(acc/16) in int16 (dst_shift -4, identity
 *     multiplier), which is undone in float before the shipped activations;
 *   - the weights and biases are the shipped ones, re-laid, not re-quantised.
 * The pre-activation therefore sees two roundings: ~s/2 per input element and
 * 16*s/256 at the output, where s is that call's quantisation step. The
 * activations (tansig_approx, sigmoid_approx, relu) are rnn.c's, verbatim.
 * The three dense layers stay as shipped (3.6 % of the network's MACs).
 *
 * Why the svdf form and not arm_nn_vec_mat_mult_t_s8: the two share one inner
 * loop; the plain s8 form stores int8, which would quantise the GRU
 * pre-activations to 8 bits (step 1/16 over +-8). The int16 store keeps the
 * output rounding two orders under the activation table's 0.04 step.
 *
 * Determinism across tiers: the integer path is identical C on x86 and
 * Cortex-M (neither CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY nor the single-
 * rounding switch is defined anywhere in the tree), so the int16 outputs are
 * bit-identical by construction; the float glue follows the image's
 * fp-contract setting like every other RNNoise row (<= 1 LSB expected).
 *
 * Quality is measured host-side on rnnoise_wav_nn8 against the oracle
 * (CLAUDE.md, Locked Decisions); the F411 image contributes cost. A labelled,
 * quality-affecting row -- never the as-shipped measurement image. */
#ifndef RNN_NN8_H
#define RNN_NN8_H
#include <stdint.h>
#include "rnn.h"

typedef struct {
   int nb_inputs;                 /* M */
   int nb_neurons;                /* N */
   const signed char *in;         /* [3N][M], gate-major rows z, r, h */
   const signed char *rec;        /* [3N][N] */
} rnnoise_nn8_gru_t;

#define RNNOISE_NN8_GRUS 3
extern const rnnoise_nn8_gru_t rnnoise_nn8_grus[RNNOISE_NN8_GRUS];   /* generated: rnn_nn8_weights.c */

/* Drop-in for rnn.c's static compute_gru(): same contract, state updated in place. */
void rnnoise_nn8_compute_gru(const GRULayer *gru, float *state, const float *input);

/* Layers the lookup failed to match (a wrong build, never a data effect; the
 * call then leaves the state untouched). Checked by the host tool. */
extern volatile uint32_t rnnoise_nn8_errors;
#ifdef RNNOISE_NN8_COUNT_SAT
/* int16 outputs that hit the +-32767 clamp -- host-side diagnostic so the
 * "never saturates" claim is counted over the corpus rather than assumed. */
extern volatile uint32_t rnnoise_nn8_saturations;
#endif
#endif
