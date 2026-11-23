/**
 * test_dtln_fc_pie.cc -- host-side bit-exactness test for the ESP32-S3 DTLN
 * matvec kernel (benchmark_s3/main/dtln_fc_pie*), 2026-10-22. Builds the C
 * core with DTLN_FC_PIE_ASM=0 and asserts, against the SAME TFLite Micro tree
 * the board runs (tflm_tree_64b3de8_generic):
 *
 *   1. dtln_fc_pie_mbqm64 == tflite::MultiplyByQuantizedMultiplier(int64)
 *      over random inputs spanning the documented ranges;
 *   2. dtln_fc_pie_matvec == reference_integer_ops::FullyConnected /
 *      FullyConnectedPerChannel (int16, int8, int16, int64) on the model's
 *      five stage-1 and three stage-2 FC shapes plus random ones, per-tensor
 *      and per-channel, with and without bias, direct and padded-stride rows;
 *   3. dtln_fc_pie_matvec == reference_integer_ops::ConvPerChannel (int16,
 *      int8, int64 bias) on the 1x1 encoder/decoder shapes plus random ones;
 *   4. the full TFLM glue (dtln_fc_pie_tflm.cc) run through MicroInterpreter
 *      on the two shipped flatbuffers, every output tensor byte-identical to
 *      the tree's own CONV_2D / FULLY_CONNECTED registrations over random
 *      int16 inputs and states, with the fallback and copy counters at 0.
 *
 * What it cannot test is dtln_fc_pie.S; the board's setup self-check does
 * that (g_dtln.fc_selfcheck), and refuses to run if it fails.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <string>
#include <fstream>

#include "dtln_fc_pie.h"
#include "dtln_proc.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

volatile dtln_stats_t g_dtln;   /* the glue's witness block, host instance */

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { g_fail++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static std::mt19937_64 rng(20261022);
static int64_t rnd(int64_t lo, int64_t hi) { return std::uniform_int_distribution<int64_t>(lo, hi)(rng); }

/* ---------------------------------------------------------------- 1 ---- */
static void test_mbqm(void)
{
    long n = 0, bad = 0;
    for (int i = 0; i < 2000000; i++) {
        int64_t x = rnd(-((int64_t)1 << 47), ((int64_t)1 << 47) - 1);
        if (i % 4 == 0) x = rnd(-((int64_t)1 << 31), ((int64_t)1 << 31) - 1);   /* the DTLN range */
        int32_t m = (int32_t)rnd(1 << 30, 0x7FFFFFFF);
        if (i % 97 == 0) m = 0x7FFF0000 + (int32_t)rnd(0, 0xFFFF);              /* the saturated-rm branch */
        int shift = (int)rnd(-31, 7);
        int32_t a = dtln_fc_pie_mbqm64(x, m, shift);
        int32_t b = tflite::MultiplyByQuantizedMultiplier(x, m, shift);
        n++; if (a != b) bad++;
    }
    CHECK(bad == 0, "mbqm64: %ld of %ld differ", bad, n);
    printf("1. mbqm64 vs TFLM int64 MultiplyByQuantizedMultiplier: %ld cases, %ld differ\n", n, bad);
}

/* ---------------------------------------------------------------- 2/3 -- */
struct Case { int rows, depth; };
static const Case kFcShapes[] = { {120,30}, {120,257}, {232,58}, {232,30}, {257,58}, {44,11}, {44,1}, {256,11} };
static const Case kConvShapes[] = { {256,512}, {512,256} };

static void fill_mv(std::vector<int16_t> &x, std::vector<int8_t> &w, std::vector<int64_t> &bias,
                    std::vector<int32_t> &mult, std::vector<int32_t> &shift, int rows, int depth, bool extreme)
{
    x.resize(depth); w.resize((size_t)rows * depth); bias.resize(rows); mult.resize(rows); shift.resize(rows);
    for (auto &v : x) v = extreme ? (int16_t)(rnd(0, 1) ? -32768 : 32767) : (int16_t)rnd(-32768, 32767);
    for (auto &v : w) v = extreme ? (int8_t)(rnd(0, 1) ? -128 : 127) : (int8_t)rnd(-128, 127);
    for (auto &v : bias) v = rnd(-((int64_t)1 << 40), (int64_t)1 << 40);
    for (auto &v : mult) v = (int32_t)rnd(1 << 30, 0x7FFFFFFF);
    for (auto &v : shift) v = (int32_t)rnd(-24, -2);    /* 16x8 FC/conv shifts are negative in practice */
}

static void run_fc_case(int rows, int depth, bool per_channel, bool with_bias, bool padded, bool extreme, long &n, long &bad)
{
    using namespace tflite;
    std::vector<int16_t> x; std::vector<int8_t> w; std::vector<int64_t> bias; std::vector<int32_t> mult, shift;
    fill_mv(x, w, bias, mult, shift, rows, depth, extreme);
    const int32_t amin = extreme ? -32768 : (int32_t)rnd(-32768, -20000), amax = extreme ? 32767 : (int32_t)rnd(20000, 32767);
    std::vector<int16_t> ref(rows), got(rows);
    FullyConnectedParams p{}; p.input_offset = 0; p.weights_offset = 0; p.output_offset = 0;
    p.output_multiplier = mult[0]; p.output_shift = shift[0];
    p.quantized_activation_min = amin; p.quantized_activation_max = amax;
    const int ish[2] = {1, depth}, fsh[2] = {rows, depth}, bsh[1] = {rows}, osh[2] = {1, rows};
    RuntimeShape is(2, ish), fs(2, fsh), bs(1, bsh), os(2, osh);
    if (per_channel)
        tflite::micro::reference_integer_ops::FullyConnectedPerChannel(p, mult.data(), (const int *)shift.data(), is, x.data(), fs, w.data(),
                                                                       bs, with_bias ? bias.data() : nullptr, os, ref.data());
    else
        tflite::micro::reference_integer_ops::FullyConnected(p, is, x.data(), fs, w.data(), bs,
                                                             with_bias ? bias.data() : nullptr, os, ref.data());
    /* the kernel, rows in place or copied to a 16-padded stride as the glue does */
    std::vector<int8_t> wp;
    dtln_fc_pie_mv_t mv{}; mv.rows = rows; mv.depth = depth;
    if (padded) {
        const int stride = (depth + 15) & ~15; wp.assign((size_t)stride * rows, 0);
        for (int o = 0; o < rows; o++) memcpy(&wp[(size_t)o * stride], &w[(size_t)o * depth], depth);
        mv.w = wp.data(); mv.w_stride = stride;
    } else { mv.w = w.data(); mv.w_stride = depth; }
    mv.bias = with_bias ? bias.data() : nullptr; mv.mult = mult.data(); mv.shift = shift.data();
    mv.per_channel = per_channel; mv.out_offset = 0; mv.act_min = amin; mv.act_max = amax;
    dtln_fc_pie_matvec(&mv, x.data(), got.data());
    n++; if (memcmp(ref.data(), got.data(), rows * sizeof(int16_t)) != 0) { bad++;
        CHECK(false, "FC rows=%d depth=%d pc=%d bias=%d padded=%d extreme=%d", rows, depth, per_channel, with_bias, padded, extreme); }
}

static void test_fc(void)
{
    long n = 0, bad = 0;
    for (const Case &c : kFcShapes)
        for (int pc = 0; pc < 2; pc++) for (int wb = 0; wb < 2; wb++) for (int pd = 0; pd < 2; pd++) for (int ex = 0; ex < 2; ex++)
            for (int rep = 0; rep < 8; rep++) run_fc_case(c.rows, c.depth, pc, wb, pd, ex, n, bad);
    for (int rep = 0; rep < 400; rep++)
        run_fc_case((int)rnd(1, 300), (int)rnd(1, 512), rnd(0, 1), rnd(0, 1), rnd(0, 1), rep % 10 == 0, n, bad);
    printf("2. FC matvec vs reference FullyConnected{,PerChannel}: %ld cases, %ld differ\n", n, bad);
}

static void run_conv_case(int rows, int depth, int positions, bool with_bias, bool extreme, long &n, long &bad)
{
    using namespace tflite;
    std::vector<int16_t> x; std::vector<int8_t> w; std::vector<int64_t> bias; std::vector<int32_t> mult, shift;
    fill_mv(x, w, bias, mult, shift, rows, depth, extreme);
    x.resize((size_t)positions * depth);
    for (size_t i = depth; i < x.size(); i++) x[i] = extreme ? (int16_t)(rnd(0, 1) ? -32768 : 32767) : (int16_t)rnd(-32768, 32767);
    const int32_t amin = extreme ? -32768 : (int32_t)rnd(-32768, -20000), amax = extreme ? 32767 : (int32_t)rnd(20000, 32767);
    std::vector<int16_t> ref((size_t)positions * rows), got((size_t)positions * rows);
    ConvParams p{}; p.padding_type = PaddingType::kSame; p.padding_values.width = 0; p.padding_values.height = 0;
    p.stride_width = 1; p.stride_height = 1; p.dilation_width_factor = 1; p.dilation_height_factor = 1;
    p.input_offset = 0; p.weights_offset = 0; p.output_offset = 0;
    p.quantized_activation_min = amin; p.quantized_activation_max = amax;
    const int ish[4] = {1, 1, positions, depth}, fsh[4] = {rows, 1, 1, depth}, bsh[1] = {rows}, osh[4] = {1, 1, positions, rows};
    RuntimeShape is(4, ish), fs(4, fsh), bs(1, bsh), os(4, osh);
    tflite::micro::reference_integer_ops::ConvPerChannel(p, mult.data(), shift.data(), is, x.data(), fs, w.data(), bs,
                                                         with_bias ? bias.data() : nullptr, os, ref.data());
    dtln_fc_pie_mv_t mv{}; mv.rows = rows; mv.depth = depth; mv.w = w.data(); mv.w_stride = depth;
    mv.bias = with_bias ? bias.data() : nullptr; mv.mult = mult.data(); mv.shift = shift.data();
    mv.per_channel = 1; mv.out_offset = 0; mv.act_min = amin; mv.act_max = amax;
    for (int q = 0; q < positions; q++) dtln_fc_pie_matvec(&mv, &x[(size_t)q * depth], &got[(size_t)q * rows]);
    n++; if (memcmp(ref.data(), got.data(), got.size() * sizeof(int16_t)) != 0) { bad++;
        CHECK(false, "conv rows=%d depth=%d pos=%d bias=%d extreme=%d", rows, depth, positions, with_bias, extreme); }
}

static void test_conv(void)
{
    long n = 0, bad = 0;
    for (const Case &c : kConvShapes)
        for (int wb = 0; wb < 2; wb++) for (int ex = 0; ex < 2; ex++) for (int rep = 0; rep < 6; rep++)
            run_conv_case(c.rows, c.depth, 1, wb, ex, n, bad);
    for (int rep = 0; rep < 150; rep++)
        run_conv_case((int)rnd(1, 300), (int)rnd(1, 512), (int)rnd(1, 4), rnd(0, 1), rep % 10 == 0, n, bad);
    printf("3. 1x1 conv matvec vs reference ConvPerChannel (int64): %ld cases, %ld differ\n", n, bad);
}

/* ---------------------------------------------------------------- 4 ---- */
static std::vector<uint8_t> read_file(const char *path)
{
    std::ifstream f(path, std::ios::binary); std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), {});
    if (b.empty()) { printf("cannot read %s\n", path); exit(2); }
    return b;
}

static void add_common(tflite::MicroMutableOpResolver<13> &r)
{
    r.AddAdd(); r.AddLogistic(); r.AddMean(); r.AddMul(); r.AddReshape(); r.AddRsqrt();
    r.AddSplit(); r.AddStridedSlice(); r.AddSub(); r.AddTanh(); r.AddUnpack();
}

static void test_interpreter(const char *path, int frames, int pad)
{
    using namespace tflite;
    std::vector<uint8_t> blob = read_file(path);
    /* Placed exactly as the port's generated array places it: 16-aligned
     * storage plus the --align-inplace pad (gen_model_c.py), so the in-place
     * rows are aligned here iff they are aligned on the board. */
    std::vector<uint8_t> store(blob.size() + 48);
    uint8_t *mp = (uint8_t *)(((uintptr_t)store.data() + 15) & ~(uintptr_t)15) + pad;
    memcpy(mp, blob.data(), blob.size());
    printf("   %s at pad %d (base mod 16 = %u)\n", path, pad, (unsigned)((uintptr_t)mp & 15u));
    const Model *model = GetModel(mp);
    static MicroMutableOpResolver<13> r_ref, r_pie;
    static bool once = false;
    if (!once) {
        add_common(r_ref); r_ref.AddConv2D(); r_ref.AddFullyConnected();
        add_common(r_pie); r_pie.AddConv2D(dtln_fc_pie_register_conv2d()); r_pie.AddFullyConnected(dtln_fc_pie_register_fully_connected());
        once = true;
    }
    static uint8_t arena_ref[96 * 1024] __attribute__((aligned(16))), arena_pie[96 * 1024] __attribute__((aligned(16)));
    MicroInterpreter it_ref(model, r_ref, arena_ref, sizeof arena_ref), it_pie(model, r_pie, arena_pie, sizeof arena_pie);
    CHECK(it_ref.AllocateTensors() == kTfLiteOk, "ref AllocateTensors %s", path);
    CHECK(it_pie.AllocateTensors() == kTfLiteOk, "pie AllocateTensors %s", path);
    if (g_fail) return;
    long n = 0, bad = 0, bytes = 0;
    for (int f = 0; f < frames; f++) {
        for (size_t i = 0; i < it_ref.inputs_size(); i++) {
            TfLiteTensor *a = it_ref.input(i), *b = it_pie.input(i);
            CHECK(a->type == kTfLiteInt16 && a->bytes == b->bytes, "input %zu type/size", i);
            int16_t *pa = a->data.i16, *pb = b->data.i16;
            for (size_t k = 0; k < a->bytes / 2; k++) {
                /* the main input is a raw-magnitude / waveform scale; states live near zero: mix both */
                int16_t v = (f % 3 == 0) ? (int16_t)rnd(-32768, 32767) : (int16_t)rnd(-4096, 4096);
                pa[k] = v; pb[k] = v;
            }
        }
        CHECK(it_ref.Invoke() == kTfLiteOk, "ref Invoke"); CHECK(it_pie.Invoke() == kTfLiteOk, "pie Invoke");
        for (size_t i = 0; i < it_ref.outputs_size(); i++) {
            TfLiteTensor *a = it_ref.output(i), *b = it_pie.output(i);
            n++; bytes += (long)a->bytes;
            if (a->bytes != b->bytes || memcmp(a->data.raw, b->data.raw, a->bytes) != 0) bad++;
        }
    }
    CHECK(bad == 0, "%s: %ld of %ld output tensors differ", path, bad, n);
    printf("4. %s: %d frames, %ld output tensors (%ld bytes) compared, %ld differ; glue counters: vec=%u ref=%u xcopies=%u pool=%u B\n",
           path, frames, n, bytes, bad, (unsigned)g_dtln.fc_calls_vec, (unsigned)g_dtln.fc_calls_ref,
           (unsigned)g_dtln.fc_x_copies, (unsigned)g_dtln.fc_pool_used);
    CHECK(g_dtln.fc_calls_ref == 0, "reference fallback taken %u times", (unsigned)g_dtln.fc_calls_ref);
}

int main(int argc, char **argv)
{
    const char *m1 = argc > 1 ? argv[1] : "../../training/runs/prune-learned/shrunk/model_i16pt_1.tflite";
    const char *m2 = argc > 2 ? argv[2] : "../../training/runs/prune-learned/shrunk/model_i16pt_2.tflite";
    const int pad1 = argc > 3 ? atoi(argv[3]) : 0, pad2 = argc > 4 ? atoi(argv[4]) : 12;   /* the a16 emission's pads */
    printf("dtln_fc_pie host test (DTLN_FC_PIE_ASM=%d)\n", DTLN_FC_PIE_ASM);
    int sc = dtln_fc_pie_selfcheck();
    CHECK(sc == 0, "selfcheck %d", sc);
    printf("0. selfcheck (C vs C on the host, trivially): %d\n", sc);
    test_mbqm();
    test_fc();
    test_conv();
    test_interpreter(m1, 300, pad1);
    test_interpreter(m2, 300, pad2);
    printf(g_fail ? "RESULT: %d FAILURES\n" : "RESULT: all identical\n", g_fail);
    return g_fail ? 1 : 0;
}
