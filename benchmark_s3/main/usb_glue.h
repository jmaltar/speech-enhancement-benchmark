#ifndef USB_GLUE_H
#define USB_GLUE_H
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
/* Installs the esp_tinyusb driver with this project's descriptors
 * (cafe:4201) and registers the wake callbacks. Call after usb_glue_abf_init(). */
void usb_glue_init(void);
/* Computes the image CRC, fills the HELLO identity and initialises the ABF1
 * CDC binding; `abf_task` is woken on every CDC rx / line-state event. */
void usb_glue_abf_init(TaskHandle_t abf_task);
#ifdef BENCH_WITH_RNNOISE
#include <stdint.h>
/* RNNoise image only: force malloc internal-first and record the free sizes
 * (called from usb_glue_abf_init), then the console witness line main.c prints
 * once per completed run: heap deltas, task stack high-water mark. */
void usb_glue_rnnoise_heap_init(void);
void usb_glue_rnnoise_witness(uint32_t frames_out, uint32_t stack_hw_bytes);
#endif
#endif
