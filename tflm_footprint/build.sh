#!/bin/bash
# Measure the TFLite Micro runtime's FLASH cost for DTLN's op set.
# Flags mirror the project's H723/F411 Release settings (-O2 -DNDEBUG,
# -ffunction-sections -fdata-sections -Wl,--gc-sections, nano.specs).
set -eu
TREE=../tflm_tree
ARCH_FLAGS="$1"; OUT="$2"; OPT="${3:--O2}"; NOPS="${4:-16}"
BUILD="obj_$OUT"; mkdir -p "$BUILD"

INC="-I$TREE -I$TREE/third_party/flatbuffers/include -I$TREE/third_party/gemmlowp \
-I$TREE/third_party/ruy -I$TREE/third_party/kissfft -I$TREE/third_party/cmsis_nn -I$TREE/third_party/cmsis_nn/Include \
-I$TREE/third_party/cmsis/CMSIS/Core/Include -I$TREE/signal/micro/kernels -I$TREE"

COMMON="$ARCH_FLAGS -mfloat-abi=hard $OPT -DNDEBUG -DTF_LITE_STRIP_ERROR_STRINGS \
-DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -DCMSIS_NN \
-fno-unwind-tables -fno-asynchronous-unwind-tables -ffunction-sections -fdata-sections \
-fmessage-length=0 -w"
CXXF="$COMMON -std=c++17 -fno-rtti -fno-exceptions -fno-threadsafe-statics"
CF="$COMMON -std=c17"

mapfile -t SRCS < <(find "$TREE" \( -name '*.cc' -o -name '*.c' \) \
  ! -name 'test_helpers*' ! -name 'mock_*' ! -name 'fake_*' | sort)
echo "compiling ${#SRCS[@]} TFLM sources at $OPT for [$ARCH_FLAGS]"
for s in "${SRCS[@]}"; do
  o="$BUILD/$(echo "${s#$TREE/}" | tr '/' '_').o"
  [ -f "$o" ] && continue
  if [[ "$s" == *.cc ]]; then arm-none-eabi-g++ $CXXF $INC -c "$s" -o "$o"
  else arm-none-eabi-gcc $CF $INC -c "$s" -o "$o"; fi
done
arm-none-eabi-gcc $CF -c startup.c -o "$BUILD/startup.o"
arm-none-eabi-g++ $CXXF -DDTLN_NOPS=$NOPS $INC -c main_tflm.cc -o "$BUILD/main_tflm.o"
arm-none-eabi-g++ $CXXF $INC -c main_base.cc -o "$BUILD/main_base.o"

LDF="$ARCH_FLAGS -mfloat-abi=hard -T link.ld --specs=nano.specs -Wl,--gc-sections -nostartfiles \
-Wl,--defsym,g_dtln_model=0x08080000"
arm-none-eabi-g++ $LDF "$BUILD/main_base.o" "$BUILD/startup.o" -o "$OUT.base.elf" -lm
arm-none-eabi-g++ $LDF "$BUILD/main_tflm.o" "$BUILD/startup.o" "$BUILD"/tensorflow_*.o "$BUILD"/signal_*.o "$BUILD"/third_party_*.o -o "$OUT.tflm.elf" -lm
