#!/usr/bin/env bash
# RPi5 tier -- the DTLN run tools (task 10 -> 16, 2026-10-26).
#
# Two executables from the classical arm's cdsp_run.c compiled unchanged against
# dtln_select (the same substitution the H723 and S3 images make), each carrying
# one model array:
#   dtln_run_p020  the pruned lambda=0.02 16x8 per-tensor model the H723 and S3 run
#                  (dtln/port/dtln_model_p020_i16pt.c; oracle dtln-prune-learned-shrunk-i16pt-tflm)
#   dtln_run_full  the full-size 16x8 export no MCU tier can hold
#                  (dtln/port/dtln_model_full_i16.c from dtln/pretrained/model_i16_*.tflite;
#                  oracle dtln-i16-tflm), arenas raised to 96 KB each (RAM is not binding here)
# TFLite Micro: the generic tree at 64b3de8 (the host wheel's commit, as on the H723/S3),
# reference kernels, built the way benchmark_s3/host_test/run.sh builds it for the host.
# dtln_proc.cc at -ffp-contract=off as on the boards; everything else at the tier's -O2.
# Usage: ./build.sh [out-dir]   (from any directory; ROOT is the project root)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
B=${1:-$HERE/build}; mkdir -p "$B"
TREE=$ROOT/tflm_tree_64b3de8_generic
DEFS="-DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON"
INC="-I$TREE -I$TREE/third_party/flatbuffers/include -I$ROOT/dtln/port -I$ROOT/harness -I$ROOT/classical"
CXX="${CXX:-g++} -std=c++17 -O2 -DNDEBUG -w -fno-exceptions -fno-rtti $DEFS $INC"
CC="${CC:-gcc} -std=c11 -O2 -DNDEBUG -w $DEFS $INC"
if [ ! -f "$B/libtflm.a" ]; then
    echo "building the TFLM tree ($(basename $TREE)) ..."
    find "$TREE/tensorflow" -name '*.cc' | grep -v -E 'test_helpers|mock_|fake_|_test\.cc$' > "$B/srcs.txt"
    : > "$B/objs.txt"
    while read -r f; do o=$B/$(echo "${f#$TREE/}" | tr '/' '_').o; $CXX -c "$f" -o "$o"; echo "$o" >> "$B/objs.txt"; done < "$B/srcs.txt"
    ar rcs "$B/libtflm.a" $(cat "$B/objs.txt")
fi
H=$ROOT/harness; C=$ROOT/classical; P=$ROOT/dtln/port
for f in $H/abf_frame.c $H/abf_file.c $H/abf_proc.c $C/cdsp.c $C/cdsp_fft_c.c $C/cdsp_select.c $P/dtln_select.c; do
    $CC -c "$f" -o "$B/$(basename "$f").o"
done
# cdsp_run.c with its registry call renamed: the one line that differs from cdsp_run
$CC -Dcdsp_select=dtln_select -c $C/cdsp_run.c -o "$B/cdsp_run_dtln.o"
$CXX -ffp-contract=off -c $P/dtln_proc.cc -o "$B/dtln_proc_p020.o"
$CXX -ffp-contract=off -DDTLN_ARENA1_BYTES=98304u -DDTLN_ARENA2_BYTES=98304u -c $P/dtln_proc.cc -o "$B/dtln_proc_full.o"
$CC -c $P/dtln_model_p020_i16pt.c -o "$B/model_p020.o"
$CC -c $P/dtln_model_full_i16.c   -o "$B/model_full.o"
OBJS="$B/abf_frame.c.o $B/abf_file.c.o $B/abf_proc.c.o $B/cdsp.c.o $B/cdsp_fft_c.c.o $B/cdsp_select.c.o $B/dtln_select.c.o $B/cdsp_run_dtln.o"
$CXX $OBJS "$B/dtln_proc_p020.o" "$B/model_p020.o" "$B/libtflm.a" -lm -o "$B/dtln_run_p020"
$CXX $OBJS "$B/dtln_proc_full.o" "$B/model_full.o" "$B/libtflm.a" -lm -o "$B/dtln_run_full"
ls -l "$B/dtln_run_p020" "$B/dtln_run_full"
