# Source this before running TensorFlow on the GPU:   source gpu_env.sh
#
# TensorFlow 2.21's wheel is built against CUDA 12.5; the CUDA 12.9 / cuDNN 9
# libraries come from the nvidia-* pip packages inside .venv, and TF's loader
# does NOT find them there on this machine ("Cannot dlopen some GPU libraries",
# no GPU listed -- 2026-10-19). Every library loads fine by path, and with the
# package lib/ directories on LD_LIBRARY_PATH the RTX 4060 is visible. The
# loader reads LD_LIBRARY_PATH at process start, so this must be set in the
# shell, not from inside Python.
_venv="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv"
_libs="$(ls -d "$_venv"/lib/python3.*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${_libs}${LD_LIBRARY_PATH:-}"
unset _venv _libs
