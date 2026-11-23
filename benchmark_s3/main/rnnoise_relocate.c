/* rnnoise_relocate.c -- the "+W" RNNoise variant on the S3: weights in internal SRAM.
 *
 * As shipped, the 88,399 B of int8 weights (rnn_data.c) are .rodata in
 * external octal flash, read through the 64 KB data cache; every weight is
 * touched once per 10 ms hop, so the whole model streams from SPI flash each
 * hop. On the STM32s flash sits on the core's own bus; here it is a serial
 * device. Whether the hop pays for that is a measured question, the same one
 * dtln_relocate.c asks for DTLN (where the answer was 0.1 %, model too large
 * for internal RAM). RNNoise's model is small enough: this file wraps
 * rnnoise_create() at the link (-Wl,--wrap=rnnoise_create, so the shared
 * rnnoise_proc.c is unmodified) and, when the caller passes NULL (the built-in
 * model), hands the library a deep copy of rnnoise_model_orig whose weight
 * arrays live in internal 8-bit-capable SRAM. g_rnnoise_model_where witnesses
 * the address of the first copied array (0x3fc... is internal DRAM;
 * 0x3c... would be flash .rodata, i.e. the copy did not happen). */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "rnnoise.h"
#include "rnn.h"
#include "rnn_data.h"

extern const struct RNNModel rnnoise_model_orig;
DenoiseState *__real_rnnoise_create(RNNModel *model);

volatile uint32_t g_rnnoise_model_where;
volatile uint32_t g_rnnoise_model_bytes;

static const rnn_weight *copy_w(const rnn_weight *src, size_t n)
{
    rnn_weight *d = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!d) return NULL;
    memcpy(d, src, n);
    if (!g_rnnoise_model_where) g_rnnoise_model_where = (uint32_t)(uintptr_t)d;
    g_rnnoise_model_bytes += (uint32_t)n;
    return d;
}

static int copy_dense(DenseLayer *dst, const DenseLayer *src)
{
    *dst = *src;
    dst->input_weights = copy_w(src->input_weights, (size_t)src->nb_inputs * (size_t)src->nb_neurons);
    dst->bias          = copy_w(src->bias, (size_t)src->nb_neurons);
    return dst->input_weights && dst->bias ? 0 : -1;
}

static int copy_gru(GRULayer *dst, const GRULayer *src)
{
    *dst = *src;
    dst->input_weights     = copy_w(src->input_weights, 3u * (size_t)src->nb_inputs * (size_t)src->nb_neurons);
    dst->recurrent_weights = copy_w(src->recurrent_weights, 3u * (size_t)src->nb_neurons * (size_t)src->nb_neurons);
    dst->bias              = copy_w(src->bias, 3u * (size_t)src->nb_neurons);
    return dst->input_weights && dst->recurrent_weights && dst->bias ? 0 : -1;
}

DenoiseState *__wrap_rnnoise_create(RNNModel *model)
{
    static RNNModel ram_model;
    static DenseLayer d_in, d_out, d_vad;
    static GRULayer g_vad, g_noise, g_denoise;
    static int done;
    if (model) return __real_rnnoise_create(model);
    if (!done) {
        const RNNModel *o = &rnnoise_model_orig;
        ram_model = *o;
        if (copy_dense(&d_in, o->input_dense) || copy_gru(&g_vad, o->vad_gru) ||
            copy_gru(&g_noise, o->noise_gru) || copy_gru(&g_denoise, o->denoise_gru) ||
            copy_dense(&d_out, o->denoise_output) || copy_dense(&d_vad, o->vad_output))
            return NULL;                      /* setup fails -> the binding NAKs; nothing silent */
        ram_model.input_dense    = &d_in;
        ram_model.vad_gru        = &g_vad;
        ram_model.noise_gru      = &g_noise;
        ram_model.denoise_gru    = &g_denoise;
        ram_model.denoise_output = &d_out;
        ram_model.vad_output     = &d_vad;
        done = 1;
    }
    return __real_rnnoise_create(&ram_model);
}
