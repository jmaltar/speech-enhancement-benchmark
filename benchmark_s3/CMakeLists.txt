# benchmark_s3 -- ESP32-S3 throughput probe (task 3, third tier).
#
# Counterpart of benchmark_f411/ and benchmark_h723/. The measurement rules are
# the same and the plain-C kernels are the same source text; what differs is the
# cycle counter, the vendor DSP library (esp-dsp, not CMSIS-DSP), and the set of
# configuration knobs that silently change what is being measured.
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(benchmark_s3)
