/**
 * bench_prune.c -- Arm B core item (c), the cost half: structured vs
 * unstructured pruning at matched weight count, on the kernel a deployed
 * int8 model would call (arm_mat_vec_mult_q7), measured 2026-10-21 on the
 * F411 against the DTLN shapes the learned thresholds actually produced.
 *
 * The 2026-10-19/20 runs pruned DTLN to 30 / 58 / 0 / 11 LSTM units, keeping
 * 55,340 of the 655,872 LSTM weights (8.44 %). Structured pruning leaves
 * SMALLER DENSE gate matrices -- rows = 4 x units, cols = the input width --
 * which the CMSIS kernel runs at full speed. Unstructured pruning at the same
 * count leaves the FULL matrix with 91.6 % zeros: a dense kernel computes the
 * zeros (its cost is the unpruned row), and the only way to skip them on a
 * core without gather SIMD is a sparse (CSR) matvec in plain C. Both are
 * measured here; "on plain C the whole effect is 1.28x" is the standing
 * reason the comparison is against CMSIS and not against a C loop.
 *
 * Buffers: the 64 KB pool holds the dense q7 matrix (seeded by the probe's
 * fixed-seed PRNG, so values are the September ones) for the CMSIS rows, and
 * is rebuilt into the CSR structure by the sparse kernel's prep -- which is
 * why the sparse kernel runs LAST. Same rules as bench_kernels.c: O(1) guard
 * returned from the timed region, O(n) checksum once, untimed.
 */
#include "bench.h"
#include "arm_math.h"
#include <string.h>

#define MAC2(r, c)   (BENCH_OPS_PER_MAC * (uint32_t)(r) * (uint32_t)(c))

static uint8_t     *s_pool;
static uint32_t     s_pool_bytes;
static const int8_t *s_x;              /* the probe's 256-entry int8 vector */
static q7_t         s_y[512];          /* rows up to 4 x 128                */
static arm_matrix_instance_q7 s_m;

static uint32_t chk_q7(const q7_t *p, uint32_t n)
{
    uint32_t acc = 0u;
    for (uint32_t i = 0; i < n; ++i) acc += (uint32_t)(int32_t)p[i];
    return acc;
}

/* --- dense CMSIS rows at the DTLN shapes ---------------------------------
 * rows = 4 x units (the stacked gates), cols = input width; numRows x numCols
 * must fit the pool. The one full-size matrix that fits is 512 x 128 (a
 * 128-unit LSTM's recurrent kernel, or LSTM1's input kernel); the 257- and
 * 256-column full matrices (131 KB) do not, and are scaled from the 512 x 128
 * and 512 x 64 rows, which measure the kernel's linearity in cols. */
static uint32_t dense(uint32_t rows, uint32_t cols)
{
    s_m.numRows = (uint16_t)rows; s_m.numCols = (uint16_t)cols;
    s_m.pData   = (q7_t *)(void *)s_pool;
    arm_mat_vec_mult_q7(&s_m, (const q7_t *)s_x, s_y);
    return (uint32_t)(int32_t)s_y[0];
}
static uint32_t k_512x128(void) { return dense(512u, 128u); }   /* full gate matrix, 128 in  */
static uint32_t k_512x64(void)  { return dense(512u, 64u);  }   /* cols linearity            */
static uint32_t k_120x30(void)  { return dense(120u, 30u);  }   /* LSTM0 recurrent, 30 units */
static uint32_t k_232x30(void)  { return dense(232u, 30u);  }   /* LSTM1 input, 30 -> 58     */
static uint32_t k_232x58(void)  { return dense(232u, 58u);  }   /* LSTM1 recurrent, 58 units */
static uint32_t k_44x256(void)  { return dense(44u, 256u);  }   /* stage-2 LSTM input, 11 u  */
static uint32_t v_512(void) { return chk_q7(s_y, 512u); }
static uint32_t v_120(void) { return chk_q7(s_y, 120u); }
static uint32_t v_232(void) { return chk_q7(s_y, 232u); }
static uint32_t v_44(void)  { return chk_q7(s_y, 44u);  }

/* --- unstructured at matched count: CSR, plain C -------------------------
 * 512 x 128 with nnz = round(0.0844 x 65,536) = 5,530 non-zeros at fixed-seed
 * random positions (the pattern unstructured magnitude pruning leaves has no
 * structure a kernel can use, so random is the honest model). Layout in the
 * pool: rowptr[513] u16, idx[nnz] u8 (cols < 256), w[nnz] q7, y[512] i32.
 * The per-element work is a gather (load idx, load x[idx], load w, MAC): the
 * dense kernel does 4 MACs per SMLAD with no index traffic at all. */
#define SP_ROWS 512u
#define SP_COLS 128u
#define SP_NNZ  5530u
static uint16_t *sp_rowptr; static uint8_t *sp_idx; static q7_t *sp_w; static int32_t *sp_y;

static uint32_t sp_rnd_state;
static uint32_t sp_rnd(void) { sp_rnd_state = sp_rnd_state * 1664525u + 1013904223u; return sp_rnd_state >> 8; }

static void prep_csr(void)
{
    uint8_t *p = s_pool;
    sp_rowptr = (uint16_t *)(void *)p; p += (SP_ROWS + 1u) * sizeof(uint16_t);
    sp_idx    = p;                     p += SP_NNZ;
    sp_w      = (q7_t *)(void *)p;     p += SP_NNZ;
    p = (uint8_t *)(((uintptr_t)p + 3u) & ~(uintptr_t)3u);
    sp_y      = (int32_t *)(void *)p;  p += SP_ROWS * sizeof(int32_t);
    /* nnz spread evenly over rows (10 or 11 per row), columns random, sorted */
    sp_rnd_state = 20261021u;
    uint32_t k = 0u;
    for (uint32_t r = 0; r < SP_ROWS; ++r) {
        sp_rowptr[r] = (uint16_t)k;
        uint32_t per = (SP_NNZ * (r + 1u)) / SP_ROWS - (SP_NNZ * r) / SP_ROWS;
        uint32_t c = 0u;
        for (uint32_t j = 0; j < per; ++j) {
            /* strictly increasing columns, spaced to fit the remaining width */
            c += 1u + sp_rnd() % ((SP_COLS - c - (per - j)) / (per - j) + 1u);
            if (c >= SP_COLS) c = SP_COLS - 1u;
            sp_idx[k] = (uint8_t)c;
            sp_w[k]   = (q7_t)((int32_t)(sp_rnd() & 0x7Fu) - 64);
            ++k;
        }
    }
    sp_rowptr[SP_ROWS] = (uint16_t)k;
}

static uint32_t k_csr(void)
{
    const int8_t *x = s_x;
    for (uint32_t r = 0; r < SP_ROWS; ++r) {
        int32_t acc = 0;
        const uint32_t e = sp_rowptr[r + 1u];
        for (uint32_t k = sp_rowptr[r]; k < e; ++k)
            acc += (int32_t)sp_w[k] * (int32_t)x[sp_idx[k]];
        sp_y[r] = acc;
    }
    return (uint32_t)sp_y[0];
}
static uint32_t v_csr(void)
{
    uint32_t acc = 0u;
    for (uint32_t i = 0; i < SP_ROWS; ++i) acc += (uint32_t)sp_y[i];
    return acc;
}

void bench_prune_run(uint32_t R, uint8_t *pool, uint32_t pool_bytes, const int8_t *x256)
{
    s_pool = pool; s_pool_bytes = pool_bytes; s_x = x256;
    if (pool_bytes < 512u * 128u) return;
    bench_measure("pr_q7_512x128", 512u*128u, MAC2(512, 128), 512u*128u, R, NULL, k_512x128, v_512);
    bench_measure("pr_q7_512x64",  512u*64u,  MAC2(512, 64),  512u*64u,  R, NULL, k_512x64,  v_512);
    bench_measure("pr_q7_120x30",  120u*30u,  MAC2(120, 30),  120u*30u,  R, NULL, k_120x30,  v_120);
    bench_measure("pr_q7_232x30",  232u*30u,  MAC2(232, 30),  232u*30u,  R, NULL, k_232x30,  v_232);
    bench_measure("pr_q7_232x58",  232u*58u,  MAC2(232, 58),  232u*58u,  R, NULL, k_232x58,  v_232);
    bench_measure("pr_q7_44x256",  44u*256u,  MAC2(44, 256),  44u*256u,  R, NULL, k_44x256,  v_44);
    /* LAST: rebuilds the pool into CSR (prep, untimed, every rep) */
    bench_measure("pr_csr_c_5530", SP_NNZ, BENCH_OPS_PER_MAC * SP_NNZ, SP_NNZ, R, prep_csr, k_csr, v_csr);
}
