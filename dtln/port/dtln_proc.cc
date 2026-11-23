/**
 * dtln_proc.cc -- DTLN 16x8 on TFLite Micro as an ABF1 processor. See the
 * header for the contract; see dtln/oracle.py for the reference this mirrors
 * line by line (enhance() for the framing, DtlnTflm for the runtime path).
 *
 * Three things are deliberate:
 *  - Boundary arithmetic (quantize / dequantize / scale) is written exactly as
 *    the oracle's numpy float32 expressions, and this file is compiled with
 *    -ffp-contract=off so no FMA contraction changes a rounding the host did
 *    not make (the one-LSB mechanism task 15 proved on the Pi).
 *  - The FFT is the classical arm's kernel seam (cdsp_fft_*), so this port
 *    shares the transform the classical port was measured with; the oracle
 *    uses numpy's float64 rfft, so the transform is the one place the two
 *    differ by construction, and task 15 measures what that costs.
 *  - Nothing allocates after setup(); the interpreters are constructed with
 *    placement new into static storage, the arenas are static, and no I/O
 *    happens anywhere in process() (harness rule 10).
 */
#include <new>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

extern "C" {
#include "dtln_proc.h"
#include "cdsp_kernels.h"

#ifdef DTLN_FC_S3_PIE
/* The ESP32-S3 PIE matvec behind FULLY_CONNECTED and 1x1 CONV_2D
 * (benchmark_s3/main/dtln_fc_pie.h, 2026-10-22). Named, guarded, witnessed. */
#include "sdkconfig.h"
#ifndef CONFIG_IDF_TARGET_ESP32S3
#error "DTLN_FC_S3_PIE names the ESP32-S3 PIE kernel; this is not an ESP32-S3 build"
#endif
#include "dtln_fc_pie.h"
#endif
extern const uint8_t dtln_model_1[];
extern const size_t  dtln_model_1_len;
extern const uint8_t dtln_model_2[];
extern const size_t  dtln_model_2_len;
#ifdef DTLN_FC_S3_PIE
/* The PIE build links the --align-inplace emission of the same flatbuffers
 * (gen_model_c.py): dtln_model_k leads with dtln_model_k_pad zero bytes so
 * the stage-2 basis rows are 16-aligned where they sit. The flatbuffer is
 * dtln_model_k + dtln_model_k_pad, and dtln_model_k_len is its length. */
extern const size_t  dtln_model_1_pad;
extern const size_t  dtln_model_2_pad;
#define DTLN_MODEL_1 (dtln_model_1 + dtln_model_1_pad)
#define DTLN_MODEL_2 (dtln_model_2 + dtln_model_2_pad)
#else
#define DTLN_MODEL_1 dtln_model_1
#define DTLN_MODEL_2 dtln_model_2
#endif
}

/* The trained geometry: fixed by the weights, not by CONFIG. */
#define DTLN_FRAME 512u
#define DTLN_HOP   128u
#define DTLN_BINS  (DTLN_FRAME / 2u + 1u)
#define DTLN_RATE  16000u
#define DTLN_MAX_STATES 8u          /* four per stage in the split layout     */
#define DTLN_MAX_STATE_LEN 128u     /* the unpruned width                     */

/* Arenas: the host runtime allocated 16,304 + 18,448 B for the lambda=0.02
 * pair; provisioned with headroom, and g_dtln reports what was used. */
#ifndef DTLN_ARENA1_BYTES
#define DTLN_ARENA1_BYTES (24u * 1024u)
#endif
#ifndef DTLN_ARENA2_BYTES
#define DTLN_ARENA2_BYTES (26u * 1024u)
#endif

volatile dtln_stats_t g_dtln;

#ifdef DTLN_DEBUG_LOG
/* Diagnostic build only (-DBENCH_DTLN_DEBUG=ON): TFLM's error strings kept and
 * appended into this buffer by the cortex_m_generic DebugLog callback, so a
 * refused graph names its reason over SWD. Runs only inside setup() -- never
 * in a timed region -- and the measurement image strips the strings. */
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
extern "C" char g_dtln_log[1024];
char g_dtln_log[1024];
static void dtln_log_cb(const char *s)
{
    size_t n = strlen(g_dtln_log), m = strlen(s);
    if (n + m + 2 > sizeof g_dtln_log) return;
    memcpy(g_dtln_log + n, s, m); g_dtln_log[n + m] = '\n'; g_dtln_log[n + m + 1] = 0;
}
#endif

namespace {

/* One interpreter with its wiring: the main tensor and the states, paired as
 * the oracle pairs them. Positions index interpreter->input(i)/output(i). */
struct Stage {
    tflite::MicroInterpreter *it;
    uint8_t  *arena;
    size_t    arena_size;
    int       in_main, out_main;                 /* positions               */
    int       n_state;
    int       in_state[DTLN_MAX_STATES];         /* positions, sorted by name */
    int       out_state[DTLN_MAX_STATES];        /* positions, sorted by ":N" */
    int       state_len[DTLN_MAX_STATES];
    float     state[DTLN_MAX_STATES][DTLN_MAX_STATE_LEN];   /* carried in float */
};

struct Ctx {
    cdsp_fft_t fft;
    float in_buf[DTLN_FRAME];
    float out_buf[DTLN_FRAME];
    float spec[DTLN_FRAME + 2u];
    float mag[DTLN_BINS];
    float mask[DTLN_BINS];
    float est[DTLN_FRAME];
    float blk[DTLN_FRAME];
    Stage s1, s2;
    int ready;
};

alignas(16) uint8_t g_arena1[DTLN_ARENA1_BYTES];
alignas(16) uint8_t g_arena2[DTLN_ARENA2_BYTES];
alignas(8)  uint8_t g_interp_storage[2][sizeof(tflite::MicroInterpreter)];
/* Exactly the op set of the shrunk 16x8 pair (dtln/port, read out of the
 * flatbuffers by tflm_footprint/opset.py): 8 in stage 1, 13 distinct overall. */
tflite::MicroMutableOpResolver<13> g_resolver;
Ctx g_ctx;

/* --- boundary arithmetic, as dtln/oracle.py writes it -------------------- */
static inline int16_t q16(float x, float s, int32_t z)
{
    float q = rintf(x / s) + (float)z;              /* np.rint(x / s) + z     */
    if (q < -32768.f) q = -32768.f;
    if (q >  32767.f) q =  32767.f;
    return (int16_t)q;
}
static inline float dq16(int16_t q, float s, int32_t z)
{
    return (float)((int32_t)q - z) * s;             /* (x - z) * s, float32   */
}

static int rank(const TfLiteTensor *t) { return t->dims->size; }
static int numel(const TfLiteTensor *t)
{
    int n = 1;
    for (int i = 0; i < t->dims->size; i++) n *= t->dims->data[i];
    return n;
}

/* Flatbuffer tensor name at an input/output position. */
static const char *tensor_name(const tflite::Model *m, bool is_input, int pos)
{
    const auto *sg = m->subgraphs()->Get(0);
    int idx = is_input ? sg->inputs()->Get(pos) : sg->outputs()->Get(pos);
    const auto *nm = sg->tensors()->Get(idx)->name();
    return nm ? nm->c_str() : "";
}

static int out_pos_tag(const char *name)
{
    /* "<call>:N" -> N; no tag -> 0, as the oracle's pos() */
    const char *c = strrchr(name, ':');
    if (!c || !c[1]) return 0;
    for (const char *p = c + 1; *p; p++) if (*p < '0' || *p > '9') return 0;
    return atoi(c + 1);
}

/* Wire one stage: main = the rank-3 tensor; states = the rest, inputs sorted
 * by name, outputs sorted by their ":N" tag; shapes must match pairwise. */
static int wire(Stage *st, const tflite::Model *m, int main_len)
{
    tflite::MicroInterpreter *it = st->it;
    st->n_state = 0; st->in_main = st->out_main = -1;
    const size_t ni = it->inputs_size(), no = it->outputs_size();
    if (ni > DTLN_MAX_STATES + 1 || no > DTLN_MAX_STATES + 1) return 1;

    const char *in_names[DTLN_MAX_STATES + 1];
    for (size_t p = 0; p < ni; p++) {
        const TfLiteTensor *t = it->input(p);
        in_names[p] = tensor_name(m, true, (int)p);
        if (rank(t) == 3) {
            if (st->in_main >= 0 || numel(t) != main_len || t->type != kTfLiteInt16) return 2;
            st->in_main = (int)p;
        }
    }
    int otag[DTLN_MAX_STATES + 1];
    for (size_t p = 0; p < no; p++) {
        const TfLiteTensor *t = it->output(p);
        otag[p] = out_pos_tag(tensor_name(m, false, (int)p));
        if (rank(t) == 3) {
            if (st->out_main >= 0 || numel(t) != main_len || t->type != kTfLiteInt16) return 3;
            st->out_main = (int)p;
        }
    }
    if (st->in_main < 0 || st->out_main < 0) return 4;

    /* selection sort of the state positions, small n */
    int ins[DTLN_MAX_STATES], nin = 0, outs[DTLN_MAX_STATES], nout = 0;
    for (size_t p = 0; p < ni; p++) if ((int)p != st->in_main) ins[nin++] = (int)p;
    for (size_t p = 0; p < no; p++) if ((int)p != st->out_main) outs[nout++] = (int)p;
    if (nin != nout) return 5;
    for (int i = 0; i < nin; i++)
        for (int j = i + 1; j < nin; j++)
            if (strcmp(in_names[ins[j]], in_names[ins[i]]) < 0) { int t = ins[i]; ins[i] = ins[j]; ins[j] = t; }
    for (int i = 0; i < nout; i++)
        for (int j = i + 1; j < nout; j++)
            if (otag[outs[j]] < otag[outs[i]]) { int t = outs[i]; outs[i] = outs[j]; outs[j] = t; }
    for (int k = 0; k < nin; k++) {
        const TfLiteTensor *ti = it->input(ins[k]), *to = it->output(outs[k]);
        if (ti->type != kTfLiteInt16 || to->type != kTfLiteInt16) return 6;
        if (numel(ti) != numel(to) || numel(ti) > (int)DTLN_MAX_STATE_LEN) return 7;
        st->in_state[k] = ins[k]; st->out_state[k] = outs[k]; st->state_len[k] = numel(ti);
    }
    st->n_state = nin;
    return 0;
}

/* Where the flatbuffer lives is a measured variable on tiers whose flash is
 * external (the S3 executes in place through a cache): a platform may
 * relocate the model before the interpreter is built. Default: in place. */
extern "C" const uint8_t *dtln_model_relocate(const uint8_t *p, size_t len)
    __attribute__((weak));
extern "C" const uint8_t *dtln_model_relocate(const uint8_t *p, size_t len) { (void)len; return p; }

static int build_stage(Stage *st, int which, const uint8_t *model_data, size_t model_len,
                       uint8_t *arena, size_t arena_size, int main_len)
{
    model_data = dtln_model_relocate(model_data, model_len);
    if (!model_data) return 8;
    const tflite::Model *m = tflite::GetModel(model_data);
    if (m->version() != TFLITE_SCHEMA_VERSION) return 8;
    st->arena = arena; st->arena_size = arena_size;
    st->it = new (g_interp_storage[which]) tflite::MicroInterpreter(m, g_resolver, arena, arena_size);
    if (st->it->AllocateTensors() != kTfLiteOk) return 9;
    return wire(st, m, main_len);
}

/* Run one stage on x[main_len] (float), states carried in st->state, exactly
 * DtlnTflm._run: quantize main and states with the INPUT params, invoke,
 * dequantize the main output and every state with the OUTPUT params. */
static int run_stage(Stage *st, const float *x, float *y, int main_len)
{
    tflite::MicroInterpreter *it = st->it;
    TfLiteTensor *tin = it->input(st->in_main);
    for (int i = 0; i < main_len; i++)
        tin->data.i16[i] = q16(x[i], tin->params.scale, tin->params.zero_point);
    for (int k = 0; k < st->n_state; k++) {
        TfLiteTensor *t = it->input(st->in_state[k]);
        for (int i = 0; i < st->state_len[k]; i++)
            t->data.i16[i] = q16(st->state[k][i], t->params.scale, t->params.zero_point);
    }
    if (it->Invoke() != kTfLiteOk) return -1;
    const TfLiteTensor *tout = it->output(st->out_main);
    for (int i = 0; i < main_len; i++)
        y[i] = dq16(tout->data.i16[i], tout->params.scale, tout->params.zero_point);
    for (int k = 0; k < st->n_state; k++) {
        const TfLiteTensor *t = it->output(st->out_state[k]);
        for (int i = 0; i < st->state_len[k]; i++)
            st->state[k][i] = dq16(t->data.i16[i], t->params.scale, t->params.zero_point);
    }
    return 0;
}

static void reset_states(Stage *st)
{
    for (int k = 0; k < st->n_state; k++)
        memset(st->state[k], 0, sizeof st->state[k]);
}

static int dtln_setup(void *c, uint32_t rate, uint32_t frame, uint32_t hop, uint32_t ch)
{
    Ctx *x = (Ctx *)c;
    g_dtln.magic = 0; g_dtln.setup_rc = 0; g_dtln.frames = 0;
    g_dtln.fc_calls_vec = g_dtln.fc_calls_ref = g_dtln.fc_x_copies = 0;
    if (rate != DTLN_RATE || frame != DTLN_FRAME || hop != DTLN_HOP || ch != 1u) {
        g_dtln.setup_rc = 1; return -1;              /* refused, never approximated */
    }
    if (!x->ready) {
        /* First CONFIG: resolver, interpreters, wiring. Idempotent after. */
#ifdef DTLN_DEBUG_LOG
        RegisterDebugLogCallback(dtln_log_cb);
#endif
        g_resolver.AddAdd();
#ifdef DTLN_FC_S3_PIE
        /* Refuse to run on a kernel that disagrees with its own C dot
         * product: the check runs once here, never in a timed region. */
        g_dtln.fc_kernel = DTLN_FC_PIE_ASM ? 1u : 2u;
        g_dtln.fc_selfcheck = (uint32_t)dtln_fc_pie_selfcheck();
        if (g_dtln.fc_selfcheck) { g_dtln.setup_rc = 40u + g_dtln.fc_selfcheck; return -1; }
        g_resolver.AddConv2D(dtln_fc_pie_register_conv2d());
        g_resolver.AddFullyConnected(dtln_fc_pie_register_fully_connected());
#else
        g_dtln.fc_kernel = 0;
        g_resolver.AddConv2D();
        g_resolver.AddFullyConnected();
#endif
        g_resolver.AddLogistic();
        g_resolver.AddMean();
        g_resolver.AddMul();
        g_resolver.AddReshape();
        g_resolver.AddRsqrt();
        g_resolver.AddSplit();
        g_resolver.AddStridedSlice();
        g_resolver.AddSub();
        g_resolver.AddTanh();
        g_resolver.AddUnpack();
        int rc;
        if ((rc = build_stage(&x->s1, 0, DTLN_MODEL_1, dtln_model_1_len, g_arena1, sizeof g_arena1, (int)DTLN_BINS)) != 0) {
            g_dtln.setup_rc = 10u + rc; return -1;
        }
        if ((rc = build_stage(&x->s2, 1, DTLN_MODEL_2, dtln_model_2_len, g_arena2, sizeof g_arena2, (int)DTLN_FRAME)) != 0) {
            g_dtln.setup_rc = 20u + rc; return -1;
        }
        if (cdsp_fft_init(&x->fft, DTLN_FRAME) != 0) { g_dtln.setup_rc = 30; return -1; }
        x->ready = 1;
    }
#ifdef DTLN_STAMP_DWT
    g_dtln.t_fft = g_dtln.t_stage1 = g_dtln.t_ifft = g_dtln.t_stage2 = g_dtln.t_ola = g_dtln.t_total = 0;
#endif
    memset(x->in_buf, 0, sizeof x->in_buf);        /* zero pre-fill = the lag */
    memset(x->out_buf, 0, sizeof x->out_buf);
    reset_states(&x->s1);
    reset_states(&x->s2);
    g_dtln.arena1_used = (uint32_t)x->s1.it->arena_used_bytes();
    g_dtln.arena2_used = (uint32_t)x->s2.it->arena_used_bytes();
    g_dtln.arena1_size = sizeof g_arena1; g_dtln.arena2_size = sizeof g_arena2;
    g_dtln.n_state1 = (uint32_t)x->s1.n_state; g_dtln.n_state2 = (uint32_t)x->s2.n_state;
    g_dtln.model1_len = (uint32_t)dtln_model_1_len; g_dtln.model2_len = (uint32_t)dtln_model_2_len;
    g_dtln.magic = 0x314C5444u;                     /* 'DTL1' */
    return 0;
}

/* One hop: exactly one iteration of oracle.enhance()'s loop plus
 * DtlnTflite.process_frame(). */
#ifdef DTLN_STAMP_DWT
extern "C" uint32_t abf_plat_ticks(void);
#define STAMP(v) do { uint32_t _t = abf_plat_ticks(); g_dtln.v += _t - t_prev; t_prev = _t; } while (0)
#else
#define STAMP(v) do { } while (0)
#endif

static int dtln_process(void *c, const int16_t *in, int16_t *out, uint32_t n)
{
    Ctx *x = (Ctx *)c;
    if (n != DTLN_HOP) return -1;
#ifdef DTLN_STAMP_DWT
    uint32_t t_prev = abf_plat_ticks(), t_start = t_prev;
#endif

    /* in_buf[:-HOP] = in_buf[HOP:]; in_buf[-HOP:] = hop / 32768 */
    memmove(x->in_buf, x->in_buf + DTLN_HOP, (DTLN_FRAME - DTLN_HOP) * sizeof(float));
    for (uint32_t i = 0; i < DTLN_HOP; i++)
        x->in_buf[DTLN_FRAME - DTLN_HOP + i] = (float)in[i] / 32768.0f;

    /* spec = rfft(frame); mag = |spec| */
    cdsp_fft_forward(&x->fft, x->in_buf, x->spec);
    for (uint32_t k = 0; k < DTLN_BINS; k++) {
        float re = x->spec[2u * k], im = x->spec[2u * k + 1u];
        x->mag[k] = sqrtf(re * re + im * im);
    }
    STAMP(t_fft);
    /* mask = stage1(mag) */
    if (run_stage(&x->s1, x->mag, x->mask, (int)DTLN_BINS) != 0) return -1;
    STAMP(t_stage1);
    /* est = irfft(mag * mask * e^{j phase}) == irfft(spec * mask) */
    for (uint32_t k = 0; k < DTLN_BINS; k++) {
        x->spec[2u * k]      *= x->mask[k];
        x->spec[2u * k + 1u] *= x->mask[k];
    }
    cdsp_fft_inverse(&x->fft, x->spec, x->est);
    STAMP(t_ifft);
    /* block = stage2(est) */
    if (run_stage(&x->s2, x->est, x->blk, (int)DTLN_FRAME) != 0) return -1;
    STAMP(t_stage2);

    /* out_buf[:-HOP] = out_buf[HOP:]; out_buf[-HOP:] = 0; out_buf += block;
     * emit out_buf[:HOP] */
    memmove(x->out_buf, x->out_buf + DTLN_HOP, (DTLN_FRAME - DTLN_HOP) * sizeof(float));
    memset(x->out_buf + (DTLN_FRAME - DTLN_HOP), 0, DTLN_HOP * sizeof(float));
    for (uint32_t i = 0; i < DTLN_FRAME; i++) x->out_buf[i] += x->blk[i];
    for (uint32_t i = 0; i < DTLN_HOP; i++) {
        float v = x->out_buf[i] * 32768.0f;         /* exact: power of two    */
        if (v < -32768.0f) v = -32768.0f;
        if (v >  32767.0f) v =  32767.0f;
        out[i] = (int16_t)rintf(v);                 /* np.rint                */
    }
    STAMP(t_ola);
#ifdef DTLN_STAMP_DWT
    g_dtln.t_total += abf_plat_ticks() - t_start;
#endif
    g_dtln.frames++;
    return 0;
}

const abf_proc_t g_dtln_proc = { "dtln-i16", &g_ctx, dtln_process, dtln_setup, NULL };

} // namespace

extern "C" const abf_proc_t *dtln_proc(void) { return &g_dtln_proc; }
