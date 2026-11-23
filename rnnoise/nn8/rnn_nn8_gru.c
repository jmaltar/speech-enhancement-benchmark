/* rnn_nn8_gru.c -- compute_gru on CMSIS-NN. Contract and rationale: rnn_nn8.h.
 * The three activation helpers below are rnn.c's, verbatim (they are static
 * there); nn8_tool.py check refuses the build if they differ. */
#include <math.h>
#include <string.h>
#include "opus_types.h"
#include "common.h"
#include "arch.h"
#include "tansig_table.h"
#include "rnn.h"
#include "rnn_nn8.h"
#include "arm_nnsupportfunctions.h"

static OPUS_INLINE float tansig_approx(float x)
{
    int i;
    float y, dy;
    float sign=1;
    /* Tests are reversed to catch NaNs */
    if (!(x<8))
        return 1;
    if (!(x>-8))
        return -1;
#ifndef FIXED_POINT
    /* Another check in case of -ffast-math */
    if (celt_isnan(x))
       return 0;
#endif
    if (x<0)
    {
       x=-x;
       sign=-1;
    }
    i = (int)floor(.5f+25*x);
    x -= .04f*i;
    y = tansig_table[i];
    dy = 1-y*y;
    y = y + x*dy*(1 - y*x);
    return sign*y;
}

static OPUS_INLINE float sigmoid_approx(float x)
{
   return .5 + .5*tansig_approx(.5*x);
}

static OPUS_INLINE float relu(float x)
{
   return x < 0 ? 0 : x;
}

volatile uint32_t rnnoise_nn8_errors;
#ifdef RNNOISE_NN8_COUNT_SAT
volatile uint32_t rnnoise_nn8_saturations;
#endif

/* Output requantisation: multiplier NN_Q31_MAX is the identity for |acc| < 2^30
 * (acc*(2^31-1)/2^31 rounds back to acc), and dst_shift -4 then stores
 * round(acc/16), half away from zero. |acc| <= 127*127*114 = 1.84 M, so /16
 * fits int16 up to 524,272 -- a pre-activation of 2048*s, past both sigmoid
 * and tanh saturation for any step s the state vectors produce. */
#define NN8_SHIFT 4
#define NN8_OUT_SCALE ((float)(1 << NN8_SHIFT))

/* Scratch is static, not stack: the F411 image reserves 0xD800 of stack for
 * the shipped 51 KB chain with ~4 KB to spare, and this is 4.2 KB. */
static int8_t  q_in[3 * MAX_NEURONS], q_st[MAX_NEURONS], q_sr[MAX_NEURONS];
static int16_t a_in[3 * MAX_NEURONS], a_st[2 * MAX_NEURONS], a_sr[MAX_NEURONS];
static float   z[MAX_NEURONS], r[MAX_NEURONS], h[MAX_NEURONS], sr[MAX_NEURONS];

static const rnnoise_nn8_gru_t *lookup(const GRULayer *gru)
{
   int k;
   for (k = 0; k < RNNOISE_NN8_GRUS; k++)
      if (rnnoise_nn8_grus[k].nb_neurons == gru->nb_neurons && rnnoise_nn8_grus[k].nb_inputs == gru->nb_inputs)
         return &rnnoise_nn8_grus[k];
   return NULL;
}

/* Symmetric int8 quantisation of one vector, scale max|x|/127. Returns the
 * step s (x ~ q*s). One divide, one multiply and lrintf per element: the same
 * IEEE operations on every tier, nothing a contraction can touch. */
static float quantize_s8(const float *x, int n, int8_t *q)
{
   int i;
   float m = 0.f, inv;
   for (i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > m) m = a; }
   if (!(m > 0.f)) { memset(q, 0, (size_t)n); return 0.f; }
   inv = 127.f / m;
   for (i = 0; i < n; i++) q[i] = (int8_t)lrintf(x[i] * inv);
   return m / 127.f;
}

static void mv(const int8_t *q, const signed char *w, int rows, int cols, int16_t *out)
{
   arm_nn_vec_mat_mult_t_svdf_s8(q, w, out, 0, 1, NN_Q31_MAX, -NN8_SHIFT, cols, rows, -32768, 32767);
#ifdef RNNOISE_NN8_COUNT_SAT
   { int i; for (i = 0; i < rows; i++) if (out[i] == 32767 || out[i] == -32768) rnnoise_nn8_saturations++; }
#endif
}

void rnnoise_nn8_compute_gru(const GRULayer *gru, float *state, const float *input)
{
   const rnnoise_nn8_gru_t *L = lookup(gru);
   int i, j;
   int M, N;
   float s_in, s_st, s_sr, k_in, k_st, k_sr;
   if (!L) { rnnoise_nn8_errors++; return; }
   M = gru->nb_inputs;
   N = gru->nb_neurons;
   s_in = quantize_s8(input, M, q_in);
   s_st = quantize_s8(state, N, q_st);
   mv(q_in, L->in, 3 * N, M, a_in);            /* z, r, h rows against the input */
   mv(q_st, L->rec, 2 * N, N, a_st);           /* z, r rows against the state */
   k_in = s_in * NN8_OUT_SCALE;
   k_st = s_st * NN8_OUT_SCALE;
   for (i = 0; i < N; i++)
   {
      /* Compute update gate. */
      float sum = gru->bias[i] + k_in * a_in[i] + k_st * a_st[i];
      z[i] = sigmoid_approx(WEIGHTS_SCALE*sum);
      /* Compute reset gate. */
      sum = gru->bias[N + i] + k_in * a_in[N + i] + k_st * a_st[N + i];
      r[i] = sigmoid_approx(WEIGHTS_SCALE*sum);
   }
   for (j = 0; j < N; j++) sr[j] = state[j] * r[j];
   s_sr = quantize_s8(sr, N, q_sr);
   mv(q_sr, L->rec + 2 * N * N, N, N, a_sr);   /* h rows against state*r */
   k_sr = s_sr * NN8_OUT_SCALE;
   for (i = 0; i < N; i++)
   {
      /* Compute output. */
      float sum = gru->bias[2*N + i] + k_in * a_in[2*N + i] + k_sr * a_sr[i];
      if (gru->activation == ACTIVATION_SIGMOID) sum = sigmoid_approx(WEIGHTS_SCALE*sum);
      else if (gru->activation == ACTIVATION_TANH) sum = tansig_approx(WEIGHTS_SCALE*sum);
      else if (gru->activation == ACTIVATION_RELU) sum = relu(WEIGHTS_SCALE*sum);
      else { rnnoise_nn8_errors++; return; }
      h[i] = z[i]*state[i] + (1-z[i])*sum;
   }
   for (i = 0; i < N; i++)
      state[i] = h[i];
}
