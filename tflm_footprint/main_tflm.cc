// Minimal DTLN-op-set TFLM link, to measure the runtime's flash cost.
// The model itself is an external symbol so none of its bytes enter this image.
#include <new>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern "C" const unsigned char g_dtln_model[];   // defined elsewhere, not linked in

#ifndef DTLN_NOPS
#define DTLN_NOPS 16
#endif

// DTLN's tensor arena. Sized by the largest of the two cores' working set;
// it is .bss, not flash, and is reported separately.
#ifndef ARENA_BYTES
#define ARENA_BYTES (45 * 1024)
#endif
alignas(16) static uint8_t g_arena[ARENA_BYTES];

static tflite::MicroMutableOpResolver<DTLN_NOPS> g_resolver;
alignas(8) static uint8_t g_interp_storage[sizeof(tflite::MicroInterpreter)];

extern "C" int main(void) {
  // Exactly the operators the shipped DTLN .tflite graphs contain, read out of
  // the flatbuffers rather than assumed: the LSTMs are NOT converted to
  // UNIDIRECTIONAL_SEQUENCE_LSTM, they are unrolled into primitives.
  g_resolver.AddAdd();
  g_resolver.AddConv2D();
  g_resolver.AddFullyConnected();
  g_resolver.AddLogistic();
  g_resolver.AddMean();
  g_resolver.AddMul();
  g_resolver.AddPack();
  g_resolver.AddPad();
  g_resolver.AddReshape();
  g_resolver.AddRsqrt();
  g_resolver.AddSplit();
  g_resolver.AddSquaredDifference();
  g_resolver.AddStridedSlice();
  g_resolver.AddSub();
  g_resolver.AddTanh();
  g_resolver.AddUnpack();
#if DTLN_NOPS >= 18
  // A full-integer conversion (what TFLM actually requires -- see below) adds
  // these two at the graph boundary.
  g_resolver.AddQuantize();
  g_resolver.AddDequantize();
#endif

  const tflite::Model* model = tflite::GetModel(g_dtln_model);
  // Placement new rather than a function-local static: a static with a
  // destructor pulls in __cxa_atexit/__dso_handle, which is runtime the real
  // firmware would not pay for and would inflate the figure being measured.
  tflite::MicroInterpreter* interpreter =
      new (g_interp_storage) tflite::MicroInterpreter(model, g_resolver,
                                                      g_arena, ARENA_BYTES);
  if (interpreter->AllocateTensors() != kTfLiteOk) return 1;
  volatile float* in = tflite::GetTensorData<float>(interpreter->input(0));
  in[0] = 0.0f;
  if (interpreter->Invoke() != kTfLiteOk) return 2;
  return (int)tflite::GetTensorData<float>(interpreter->output(0))[0];
}
