#!/usr/bin/env bash
# Build and run the host bit-exactness test for the S3 DTLN matvec kernel
# against the TFLite Micro tree the board runs (64b3de8, generic). The core is
# built with DTLN_FC_PIE_ASM=0 (its C dot product); the assembly is verified on
# the board by the setup self-check. Usage: ./run.sh  (from this directory)
set -euo pipefail
cd "$(dirname "$0")"
TREE=../../tflm_tree_64b3de8_generic
B=build; mkdir -p $B
DEFS="-DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -DDTLN_FC_PIE_ASM=0"
INC="-I$TREE -I$TREE/third_party/flatbuffers/include -I../main -I../../dtln/port -I../../harness -I../../classical"
CXX="${CXX:-g++} -std=c++17 -O2 -w -fno-exceptions -fno-rtti $DEFS $INC"
CC="${CC:-gcc} -std=c11 -O2 -Wall -Wextra $DEFS $INC"
# the tree, once (header-only flatbuffers; signal/ and third_party sources are not needed)
if [ ! -f $B/libtflm_host.a ]; then
    echo "building the TFLM tree for the host ..."
    find $TREE/tensorflow -name '*.cc' | grep -v -E 'test_helpers|mock_|fake_|_test\.cc$' > $B/srcs.txt
    : > $B/objs.txt
    while read -r f; do o=$B/$(echo "$f" | tr '/' '_' ).o; $CXX -c "$f" -o "$o"; echo "$o" >> $B/objs.txt; done < $B/srcs.txt
    ar rcs $B/libtflm_host.a $(cat $B/objs.txt)
fi
$CC  -c ../main/dtln_fc_pie_core.c   -o $B/core.o
$CXX -c ../main/dtln_fc_pie_tflm.cc  -o $B/glue.o
$CXX -c test_dtln_fc_pie.cc          -o $B/test.o
$CXX $B/test.o $B/glue.o $B/core.o $B/libtflm_host.a -o $B/test_dtln_fc_pie
# the pads the PIE build links (dtln/port/dtln_model_p020_i16pt_a16.c), read from the emission
A16=../../dtln/port/dtln_model_p020_i16pt_a16.c
M=../../training/runs/prune-learned/shrunk
PAD1=$(sed -n 's/^const size_t dtln_model_1_pad = \([0-9]*\);/\1/p' $A16)
PAD2=$(sed -n 's/^const size_t dtln_model_2_pad = \([0-9]*\);/\1/p' $A16)
./$B/test_dtln_fc_pie $M/model_i16pt_1.tflite $M/model_i16pt_2.tflite "${PAD1:-0}" "${PAD2:-0}" "$@"
