/* HOST TEST ONLY (benchmark_s3/host_test/run_wav.sh). dtln_proc.cc refuses
 * DTLN_FC_S3_PIE outside an ESP32-S3 firmware build by checking the IDF
 * target in sdkconfig.h; this stub satisfies that check so the SAME
 * processor source can be linked on the host with the PIE glue and its C
 * dot product. Never on the include path of a firmware build. */
#define CONFIG_IDF_TARGET_ESP32S3 1
