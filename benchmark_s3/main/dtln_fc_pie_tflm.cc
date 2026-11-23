/**
 * dtln_fc_pie_tflm.cc -- TFLite Micro registrations that route FULLY_CONNECTED
 * and 1x1 CONV_2D (int16 in, int8 weights, int64 bias) through the PIE matvec
 * in dtln_fc_pie_core.c / dtln_fc_pie.S. See dtln_fc_pie.h for the contract.
 *
 * What this file decides, and witnesses in g_dtln:
 *  - a node takes the fast path only if its types, zero points, geometry and
 *    bias type match the int64 reference branch exactly; anything else runs
 *    the reference kernel from the same tree and counts in fc_calls_ref;
 *  - weight rows that are not 16-byte aligned (stage 1's 30/257/58-wide rows,
 *    the mask Dense) are copied once at Prepare into a static 16-aligned pool
 *    with zero-padded rows (fc_pool_used / fc_pool_addr); rows that are
 *    aligned where they sit (the stage-2 basis, 512- and 256-wide, in flash
 *    or PSRAM) are used in place, so model placement stays a measured
 *    variable;
 *  - an input vector that is not 16-byte aligned is copied into a static
 *    scratch (fc_x_copies); every TFLM arena tensor is 16-aligned, so the
 *    count is expected to be 0 and is reported either way.
 * Nothing here allocates after Prepare and nothing in Eval does I/O.
 */
#include "dtln_fc_pie.h"
#include "dtln_proc.h"
#include <string.h>
#include <stdint.h>

#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/micro/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/micro/micro_log.h"

#ifndef DTLN_FC_PIE_POOL_BYTES
#define DTLN_FC_PIE_POOL_BYTES (84u * 1024u)   /* stage 1 + mask Dense pad to 80,704 B */
#endif
#ifndef DTLN_FC_PIE_MAX_DEPTH
#define DTLN_FC_PIE_MAX_DEPTH 1024u
#endif

namespace {

__attribute__((aligned(16))) int8_t  s_pool[DTLN_FC_PIE_POOL_BYTES];
__attribute__((aligned(16))) int16_t s_xscratch[DTLN_FC_PIE_MAX_DEPTH];
size_t s_pool_used = 0;

bool aligned16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

/* Make rows usable by the aligned 128-bit loads: in place if every row start
 * is 16-aligned, else a zero-padded copy in the pool. Returns false if the
 * pool is exhausted (Prepare then fails: the board refuses rather than
 * measuring a different thing). */
bool place_weights(dtln_fc_pie_mv_t *mv, const int8_t *w, int rows, int depth)
{
    mv->rows = rows; mv->depth = depth;
    if (aligned16(w) && (depth % 16 == 0 || rows == 1)) {
        mv->w = w; mv->w_stride = depth;
        return true;
    }
    const int stride = (depth + 15) & ~15;
    const size_t need = (size_t)stride * (size_t)rows;
    if (s_pool_used + need > sizeof s_pool) return false;
    int8_t *dst = s_pool + s_pool_used;
    for (int o = 0; o < rows; o++) {
        memcpy(dst + (size_t)o * stride, w + (size_t)o * depth, (size_t)depth);
        memset(dst + (size_t)o * stride + depth, 0, (size_t)(stride - depth));
    }
    s_pool_used += need;
    g_dtln.fc_pool_used = (uint32_t)s_pool_used;
    mv->w = dst; mv->w_stride = stride;
    return true;
}

const int16_t *aligned_x(const int16_t *x, int depth)
{
    if (aligned16(x)) return x;
    g_dtln.fc_x_copies++;
    memcpy(s_xscratch, x, (size_t)depth * sizeof(int16_t));
    return s_xscratch;
}

/* ---------------------------------------------------------------- FC ---- */

struct FcData {
    tflite::micro::OpDataFullyConnected base;   /* first: the reference helpers cast to it */
    dtln_fc_pie_mv_t mv;
    int32_t mult1, shift1;                      /* per-tensor copies (OpData's shift is a plain int) */
    bool fast;
};

void *FcInit(TfLiteContext *context, const char *, size_t)
{
    return context->AllocatePersistentBuffer(context, sizeof(FcData));
}

TfLiteStatus FcPrepare(TfLiteContext *context, TfLiteNode *node)
{
    using namespace tflite::micro;
    MicroContext *mc = GetMicroContext(context);
    FcData *d = static_cast<FcData *>(node->user_data);
    const auto *params = static_cast<const TfLiteFullyConnectedParams *>(node->builtin_data);
    TfLiteTensor *input  = mc->AllocateTempInputTensor(node, kFullyConnectedInputTensor);
    TfLiteTensor *filter = mc->AllocateTempInputTensor(node, kFullyConnectedWeightsTensor);
    TfLiteTensor *bias   = mc->AllocateTempInputTensor(node, kFullyConnectedBiasTensor);
    TfLiteTensor *output = mc->AllocateTempOutputTensor(node, kFullyConnectedOutputTensor);
    TF_LITE_ENSURE(context, input != nullptr && filter != nullptr && output != nullptr);
    TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
    if (input->type != kTfLiteInt16 || filter->type != kTfLiteInt8) {
        MicroPrintf("dtln_fc_pie: FC input %s / filter %s not supported",
                    TfLiteTypeGetName(input->type), TfLiteTypeGetName(filter->type));
        return kTfLiteError;
    }
    TF_LITE_ENSURE_OK(context, CalculateOpDataFullyConnected(context, params->activation, input->type,
                                                             input, filter, bias, output, &d->base));
    const bool int64_acc = (bias != nullptr) ? (bias->type == kTfLiteInt64)
                                             : (params->quantized_bias_type != kTfLiteInt32);
    d->fast = int64_acc && d->base.input_zero_point == 0 && d->base.filter_zero_point == 0 &&
              filter->dims->size == 2 && filter->dims->data[1] <= (int)DTLN_FC_PIE_MAX_DEPTH;
    if (d->fast) {
        dtln_fc_pie_mv_t *mv = &d->mv;
        if (!place_weights(mv, filter->data.int8, filter->dims->data[0], filter->dims->data[1])) {
            MicroPrintf("dtln_fc_pie: weight pool exhausted");
            return kTfLiteError;
        }
        mv->bias = bias ? bias->data.i64 : nullptr;
        mv->per_channel = d->base.is_per_channel ? 1 : 0;
        if (d->base.is_per_channel) {
            mv->mult = d->base.per_channel_output_multiplier;
            mv->shift = d->base.per_channel_output_shift;
        } else {
            d->mult1 = d->base.output_multiplier;
            d->shift1 = (int32_t)d->base.output_shift;
            mv->mult = &d->mult1;
            mv->shift = &d->shift1;
        }
        mv->out_offset = d->base.output_zero_point;
        mv->act_min = d->base.output_activation_min;
        mv->act_max = d->base.output_activation_max;
    }
    mc->DeallocateTempTfLiteTensor(input);
    mc->DeallocateTempTfLiteTensor(filter);
    if (bias) mc->DeallocateTempTfLiteTensor(bias);
    mc->DeallocateTempTfLiteTensor(output);
    return kTfLiteOk;
}

TfLiteStatus FcEval(TfLiteContext *context, TfLiteNode *node)
{
    using namespace tflite::micro;
    const FcData *d = static_cast<const FcData *>(node->user_data);
    const TfLiteEvalTensor *input  = GetEvalInput(context, node, kFullyConnectedInputTensor);
    const TfLiteEvalTensor *filter = GetEvalInput(context, node, kFullyConnectedWeightsTensor);
    const TfLiteEvalTensor *bias   = (NumInputs(node) == 3) ? GetEvalInput(context, node, kFullyConnectedBiasTensor) : nullptr;
    TfLiteEvalTensor *output = GetEvalOutput(context, node, kFullyConnectedOutputTensor);
    if (d->fast) {
        const dtln_fc_pie_mv_t *mv = &d->mv;
        const int16_t *x = GetTensorData<int16_t>(input);
        int16_t *y = GetTensorData<int16_t>(output);
        const int batches = GetTensorShape(input).FlatSize() / mv->depth;
        for (int b = 0; b < batches; b++)
            dtln_fc_pie_matvec(mv, aligned_x(x + (size_t)b * mv->depth, mv->depth), y + (size_t)b * mv->rows);
        g_dtln.fc_calls_vec++;
        return kTfLiteOk;
    }
    /* Everything the fast path declined: the reference kernels of this tree,
     * int16 x int8 only (the model has nothing else), counted. */
    g_dtln.fc_calls_ref++;
    const auto *params = static_cast<const TfLiteFullyConnectedParams *>(node->builtin_data);
    const bool int32_acc = (bias != nullptr && bias->type == kTfLiteInt32) ||
                           (bias == nullptr && params->quantized_bias_type == kTfLiteInt32);
    if (int32_acc) {
        if (d->base.is_per_channel)
            reference_integer_ops::FullyConnectedPerChannel(
                FullyConnectedParamsQuantized(d->base), d->base.per_channel_output_multiplier,
                reinterpret_cast<const int *>(d->base.per_channel_output_shift),
                GetTensorShape(input), GetTensorData<int16_t>(input), GetTensorShape(filter),
                GetTensorData<int8_t>(filter), GetTensorShape(bias), GetOptionalTensorData<int32_t>(bias),
                GetTensorShape(output), GetTensorData<int16_t>(output));
        else
            reference_integer_ops::FullyConnected(
                FullyConnectedParamsQuantized(d->base), GetTensorShape(input), GetTensorData<int16_t>(input),
                GetTensorShape(filter), GetTensorData<int8_t>(filter), GetTensorShape(bias),
                GetOptionalTensorData<int32_t>(bias), GetTensorShape(output), GetTensorData<int16_t>(output));
    } else {
        if (d->base.is_per_channel)
            reference_integer_ops::FullyConnectedPerChannel(
                FullyConnectedParamsQuantized(d->base), d->base.per_channel_output_multiplier,
                reinterpret_cast<const int *>(d->base.per_channel_output_shift),
                GetTensorShape(input), GetTensorData<int16_t>(input), GetTensorShape(filter),
                GetTensorData<int8_t>(filter), GetTensorShape(bias), GetOptionalTensorData<int64_t>(bias),
                GetTensorShape(output), GetTensorData<int16_t>(output));
        else
            reference_integer_ops::FullyConnected(
                FullyConnectedParamsQuantized(d->base), GetTensorShape(input), GetTensorData<int16_t>(input),
                GetTensorShape(filter), GetTensorData<int8_t>(filter), GetTensorShape(bias),
                GetOptionalTensorData<int64_t>(bias), GetTensorShape(output), GetTensorData<int16_t>(output));
    }
    return kTfLiteOk;
}

/* ------------------------------------------------------------ CONV_2D ---- */

struct ConvData {
    tflite::micro::OpDataConv base;   /* first: ConvPrepare casts user_data to it */
    dtln_fc_pie_mv_t mv;
    int positions;                    /* batch * out_h * out_w                */
    bool fast;
};

void *ConvInitPie(TfLiteContext *context, const char *, size_t)
{
    return context->AllocatePersistentBuffer(context, sizeof(ConvData));
}

TfLiteStatus ConvPreparePie(TfLiteContext *context, TfLiteNode *node)
{
    using namespace tflite::micro;
    TF_LITE_ENSURE_OK(context, ConvPrepare(context, node));     /* the tree's own: multipliers, padding, ranges */
    MicroContext *mc = GetMicroContext(context);
    ConvData *d = static_cast<ConvData *>(node->user_data);
    const auto *params = static_cast<const TfLiteConvParams *>(node->builtin_data);
    TfLiteTensor *input  = mc->AllocateTempInputTensor(node, kConvInputTensor);
    TfLiteTensor *filter = mc->AllocateTempInputTensor(node, kConvWeightsTensor);
    TfLiteTensor *bias   = mc->AllocateTempInputTensor(node, kConvBiasTensor);
    TfLiteTensor *output = mc->AllocateTempOutputTensor(node, kConvOutputTensor);
    TF_LITE_ENSURE(context, input != nullptr && filter != nullptr && output != nullptr);
    d->fast = false;
    const bool int64_acc = (bias != nullptr) ? (bias->type == kTfLiteInt64)
                                             : (params->quantized_bias_type != kTfLiteInt32);
    if (input->type == kTfLiteInt16 && filter->type == kTfLiteInt8 && output->type == kTfLiteInt16 &&
        int64_acc && input->dims->size == 4 && filter->dims->size == 4 && output->dims->size == 4 &&
        filter->dims->data[1] == 1 && filter->dims->data[2] == 1 &&
        filter->dims->data[3] == input->dims->data[3] &&            /* groups == 1 */
        filter->dims->data[0] == output->dims->data[3] &&
        params->stride_width == 1 && params->stride_height == 1 &&
        params->dilation_width_factor == 1 && params->dilation_height_factor == 1 &&
        d->base.padding.width == 0 && d->base.padding.height == 0 &&
        input->dims->data[0] == output->dims->data[0] && input->dims->data[1] == output->dims->data[1] &&
        input->dims->data[2] == output->dims->data[2] &&
        d->base.input_zero_point == 0 && d->base.filter_zero_point == 0 &&
        input->dims->data[3] <= (int)DTLN_FC_PIE_MAX_DEPTH) {
        dtln_fc_pie_mv_t *mv = &d->mv;
        if (!place_weights(mv, filter->data.int8, filter->dims->data[0], filter->dims->data[3])) {
            MicroPrintf("dtln_fc_pie: weight pool exhausted");
            return kTfLiteError;
        }
        mv->bias = bias ? bias->data.i64 : nullptr;
        mv->per_channel = 1;                         /* conv is per-channel by the spec */
        mv->mult = d->base.per_channel_output_multiplier;
        mv->shift = d->base.per_channel_output_shift;
        mv->out_offset = 0;                          /* the int16 conv reference adds none */
        mv->act_min = d->base.output_activation_min;
        mv->act_max = d->base.output_activation_max;
        d->positions = input->dims->data[0] * input->dims->data[1] * input->dims->data[2];
        d->fast = true;
    }
    mc->DeallocateTempTfLiteTensor(input);
    mc->DeallocateTempTfLiteTensor(filter);
    if (bias) mc->DeallocateTempTfLiteTensor(bias);
    mc->DeallocateTempTfLiteTensor(output);
    return kTfLiteOk;
}

TfLiteStatus ConvEvalPie(TfLiteContext *context, TfLiteNode *node)
{
    using namespace tflite::micro;
    const ConvData *d = static_cast<const ConvData *>(node->user_data);
    const TfLiteEvalTensor *input  = GetEvalInput(context, node, kConvInputTensor);
    const TfLiteEvalTensor *filter = GetEvalInput(context, node, kConvWeightsTensor);
    const TfLiteEvalTensor *bias   = (NumInputs(node) == 3) ? GetEvalInput(context, node, kConvBiasTensor) : nullptr;
    TfLiteEvalTensor *output = GetEvalOutput(context, node, kConvOutputTensor);
    if (d->fast) {
        const dtln_fc_pie_mv_t *mv = &d->mv;
        const int16_t *x = GetTensorData<int16_t>(input);
        int16_t *y = GetTensorData<int16_t>(output);
        for (int p = 0; p < d->positions; p++)
            dtln_fc_pie_matvec(mv, aligned_x(x + (size_t)p * mv->depth, mv->depth), y + (size_t)p * mv->rows);
        g_dtln.fc_calls_vec++;
        return kTfLiteOk;
    }
    g_dtln.fc_calls_ref++;
    const auto &params = *static_cast<const TfLiteConvParams *>(node->builtin_data);
    if (input->type != kTfLiteInt16 || filter->type != kTfLiteInt8) {
        MicroPrintf("dtln_fc_pie: conv input %s / filter %s not supported",
                    TfLiteTypeGetName(input->type), TfLiteTypeGetName(filter->type));
        return kTfLiteError;
    }
    const bool int32_acc = (bias != nullptr && bias->type == kTfLiteInt32) ||
                           (bias == nullptr && params.quantized_bias_type == kTfLiteInt32);
    if (int32_acc)
        reference_integer_ops::ConvPerChannel(
            ConvParamsQuantized(params, d->base), d->base.per_channel_output_multiplier,
            d->base.per_channel_output_shift, GetTensorShape(input), GetTensorData<int16_t>(input),
            GetTensorShape(filter), GetTensorData<int8_t>(filter), GetTensorShape(bias),
            GetOptionalTensorData<std::int32_t>(bias), GetTensorShape(output), GetTensorData<int16_t>(output));
    else
        reference_integer_ops::ConvPerChannel(
            ConvParamsQuantized(params, d->base), d->base.per_channel_output_multiplier,
            d->base.per_channel_output_shift, GetTensorShape(input), GetTensorData<int16_t>(input),
            GetTensorShape(filter), GetTensorData<int8_t>(filter), GetTensorShape(bias),
            GetOptionalTensorData<std::int64_t>(bias), GetTensorShape(output), GetTensorData<int16_t>(output));
    return kTfLiteOk;
}

} // namespace

TFLMRegistration dtln_fc_pie_register_fully_connected(void)
{
    g_dtln.fc_pool_addr = (uint32_t)(uintptr_t)s_pool;
    g_dtln.fc_pool_size = (uint32_t)sizeof s_pool;
    return tflite::micro::RegisterOp(FcInit, FcPrepare, FcEval);
}

TFLMRegistration dtln_fc_pie_register_conv2d(void)
{
    return tflite::micro::RegisterOp(ConvInitPie, ConvPreparePie, ConvEvalPie);
}
