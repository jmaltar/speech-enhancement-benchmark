#!/usr/bin/env bash
# Utterance-level host check for the S3 PIE glue: the port's own processor
# (dtln/port/dtln_proc.cc, cdsp_fft_c transform, --align-inplace model
# array) linked twice -- TFLM reference registrations vs the PIE glue with
# its C dot product -- run on real VoiceBank utterances through the file
# binding. Asserts the two outputs are byte-identical, and reports each
# against the Python TFLM oracle (enhanced/dtln-prune-learned-shrunk-i16pt-
# tflm), where a few LSB is the float32-FFT gap the board also shows.
# Usage: ./run_wav.sh [N utterances, default 4]   (from this directory; needs run.sh's tree build)
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-4}
TREE=../../tflm_tree_64b3de8_generic
B=build; mkdir -p $B/wav
[ -f $B/libtflm_host.a ] || { echo "run ./run.sh first (builds the TFLM tree)"; exit 1; }
COMMON="-DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON"
INC="-I$TREE -I$TREE/third_party/flatbuffers/include -I../main -I../../dtln/port -I../../harness -I../../classical"
CXX="${CXX:-g++} -std=c++17 -O2 -w -fno-exceptions -fno-rtti $COMMON $INC"
CC="${CC:-gcc} -std=c11 -O2 -w $COMMON $INC"
H=../../harness; C=../../classical; P=../../dtln/port
for f in $H/abf_frame.c $H/abf_file.c $H/abf_proc.c $H/abf_wav.c $C/cdsp.c $C/cdsp_fft_c.c $C/cdsp_select.c $P/dtln_select.c; do
    $CC -c $f -o $B/wav/$(basename $f).o
done
$CXX -c dtln_wav_host.cc -o $B/wav/main.o
# ref: the plain model array, the tree's registrations
$CC  -c $P/dtln_model_p020_i16pt.c -o $B/wav/model_ref.o
$CXX -ffp-contract=off -c $P/dtln_proc.cc -o $B/wav/proc_ref.o
$CXX $B/wav/main.o $B/wav/proc_ref.o $B/wav/model_ref.o $B/wav/*.c.o $B/libtflm_host.a -o $B/dtln_wav_ref
# piec: the --align-inplace array, the PIE glue, C dot product (stub sdkconfig.h: host only)
$CC  -c $P/dtln_model_p020_i16pt_a16.c -o $B/wav/model_a16.o
$CC  -DDTLN_FC_PIE_ASM=0 -c ../main/dtln_fc_pie_core.c -o $B/wav/core.o
$CXX -DDTLN_FC_PIE_ASM=0 -c ../main/dtln_fc_pie_tflm.cc -o $B/wav/glue.o
$CXX -ffp-contract=off -DDTLN_FC_S3_PIE=1 -DDTLN_FC_PIE_ASM=0 -Istub -c $P/dtln_proc.cc -o $B/wav/proc_pie.o
$CXX $B/wav/main.o $B/wav/proc_pie.o $B/wav/model_a16.o $B/wav/core.o $B/wav/glue.o $B/wav/*.c.o $B/libtflm_host.a -o $B/dtln_wav_piec
ROOT=${BENCHMARK_DATA_ROOT:-/mnt/data/benchmark-data}
ORACLE=$ROOT/enhanced/dtln-prune-learned-shrunk-i16pt-tflm/voicebank16k
../../.venv/bin/python - "$ROOT/manifest.csv" "$N" > $B/wav/list.txt <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['corpus'] == 'voicebank16k' and r['role'] == 'eval']
# spread over the corpus, not its head (trap 14): every len/N-th utterance
n = int(sys.argv[2]); step = max(1, len(rows) // n)
for r in rows[::step][:n]: print(r['id'], r['noisy_path'])
PY
fail=0
while read -r id noisy; do
    ./$B/dtln_wav_ref  --in "$noisy" --out $B/wav/$id.ref.wav  > $B/wav/$id.ref.log
    ./$B/dtln_wav_piec --in "$noisy" --out $B/wav/$id.piec.wav > $B/wav/$id.piec.log
    if cmp -s $B/wav/$id.ref.wav $B/wav/$id.piec.wav; then same=identical; else same=DIFFER; fail=1; fi
    ../../.venv/bin/python - $B/wav/$id.piec.wav $ORACLE/$id.wav "$id" "$same" "$(grep -o 'fc_kernel=.*B' $B/wav/$id.piec.log)" <<'PY'
import sys, numpy as np, wave
def rd(p):
    with wave.open(p) as w: return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.int64)
a, b = rd(sys.argv[1]), rd(sys.argv[2]); n = min(len(a), len(b))
d = a[:n] - b[:n]
nmse = 10 * np.log10((d.astype(float) ** 2).sum() / max(1.0, (b[:n].astype(float) ** 2).sum()))
print(f"{sys.argv[3]}: ref vs piec {sys.argv[4]}; piec vs Python TFLM oracle: max|d| {int(np.abs(d).max())} LSB, {int((d != 0).sum())} of {n} samples differ, NMSE {nmse:.1f} dB  [{sys.argv[5]}]")
PY
done < $B/wav/list.txt
[ $fail -eq 0 ] && echo "RESULT: ref and piec byte-identical on every utterance" || { echo "RESULT: ref and piec DIFFER"; exit 1; }
