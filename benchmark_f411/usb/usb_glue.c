/**
 * usb_glue.c -- board-side bring-up for TinyUSB on the WeAct STM32F411, and
 * the identity handed to the ABF1 CDC binding.
 *
 * TinyUSB's dwc2 port drives the OTG_FS registers directly; it does NOT use
 * ST's HAL PCD driver and it does not configure the peripheral's clock, pins
 * or interrupt. Those three things are this file's first job.
 *
 * CubeMX still generates MX_USB_OTG_FS_PCD_Init() in Core/Src/usb_otg.c. It is
 * deliberately NOT called from main.c: HAL_PCD_Init() would reset and
 * reconfigure the same core registers TinyUSB is about to take over, and the
 * two drivers cannot both own the peripheral. The generated file is left in
 * place so a CubeMX regeneration does not fight this project; --gc-sections
 * drops the unused code.
 *
 * The placeholder echo that used to live here (264 B of flash, proved the pipe
 * at 376 KiB/s) is gone: the data path is now abf_cdc_service(), task 7's
 * framing, called from the main loop beside tud_task(). It keeps the one
 * property the echo had to hand on -- it never blocks.
 */
#include "usb_glue.h"
#include "main.h"
#include "tusb.h"
#include "abf_cdc.h"
#include "abf_proc.h"
#include "cdsp.h"
#ifdef BENCH_WITH_RNNOISE
#include "rnnoise_proc.h"
#endif

#if !defined(BENCH_WITH_RNNOISE) && !defined(BENCH_WITH_CDSP)
/* The build-twice twin of the RNNoise image (BENCH_WITH_CDSP=OFF): binding,
 * bypass and OLA only. Registry semantics as cdsp_select's first lines. */
static const abf_proc_t *basic_select(const abf_config_t *cfg)
{
    if (cfg->flags & ABF_CF_BYPASS) return abf_proc_bypass();
    if (cfg->algo_id >> ABF_NOISE_SHIFT) return NULL;
    uint32_t algo = cfg->algo_id & ABF_ALGO_MASK;
    if (algo == ABF_ALGO_BYPASS) return abf_proc_bypass();
    if (algo == ABF_ALGO_OLA)    return abf_proc_ola();
    return NULL;
}
#endif

void usb_glue_init(void)
{
    GPIO_InitTypeDef gpio = {0};

    __HAL_RCC_GPIOA_CLK_ENABLE();

    /* PA11 = OTG_FS_DM, PA12 = OTG_FS_DP. VERY_HIGH speed is not optional --
     * full-speed edges do not meet the eye diagram at lower drive settings. */
    gpio.Pin       = GPIO_PIN_11 | GPIO_PIN_12;
    gpio.Mode      = GPIO_MODE_AF_PP;
    gpio.Pull      = GPIO_NOPULL;
    gpio.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;
    gpio.Alternate = GPIO_AF10_OTG_FS;
    HAL_GPIO_Init(GPIOA, &gpio);

    __HAL_RCC_USB_OTG_FS_CLK_ENABLE();

    /* Priority 5 leaves room above it if the audio path later needs a higher
     * -priority DMA or timer interrupt. Recorded because interrupt priority is
     * part of the jitter measurement's configuration, not an implementation
     * detail. */
    HAL_NVIC_SetPriority(OTG_FS_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(OTG_FS_IRQn);

    tusb_rhport_init_t init = {
        .role  = TUSB_ROLE_DEVICE,
        .speed = TUSB_SPEED_FULL,
    };
    tusb_init(BOARD_TUD_RHPORT, &init);
}

/* ---- identity ------------------------------------------------------------
 *
 * HELLO.image_crc32 is the CRC-32 of the flash image as it actually sits in
 * flash -- vector table through the end of .data's load image, which is
 * exactly the span `objcopy -O binary` emits (with --gap-fill 0xFF, matching
 * erased flash). The host CRCs the .bin and compares. This is verify_image at
 * the protocol level: a stream from a foreign or stale image cannot be
 * mistaken for a run of the build in the tree, which both boards have already
 * demonstrated is a real failure mode. Computed once at boot (~4 ms at
 * ~8 cycles/byte for a 50 KB image) with the same CRC the framing uses, so
 * there is no second implementation to disagree.
 */
extern uint32_t _sidata, _sdata, _edata;   /* from STM32F411xx_FLASH.ld */

static uint32_t image_crc32(void)
{
    const uint8_t *start = (const uint8_t *)FLASH_BASE;
    const uint8_t *end   = (const uint8_t *)&_sidata
                         + ((uintptr_t)&_edata - (uintptr_t)&_sdata);
    uint32_t crc = abf_crc32(ABF_CRC_INIT, start, (size_t)(end - start));
    return abf_crc32_final(crc);
}

#ifndef ABF_BUILD_ID
#  define ABF_BUILD_ID "unstamped"
#endif

void usb_glue_abf_init(void)
{
    abf_cdc_ident_t id = {
        .platform    = "STM32F411",
        .build_id    = ABF_BUILD_ID,
        .image_crc32 = image_crc32(),
        .cpu_hz      = SystemCoreClock,
    };
    /* Bypass is the processor of record until a CONFIG says otherwise; the
     * selector then picks bypass, OLA or one of task 8's three classical gain
     * rules from algo_id, so this one image serves every algorithm it carries
     * and the run configuration travels with the run. cdsp.c is compiled
     * from ../classical unmodified, with cdsp_fft_cmsis.c under it. */
    abf_cdc_init(abf_proc_bypass(), &id);
#if defined(BENCH_WITH_RNNOISE)
    abf_cdc_set_selector(rnnoise_select);   /* RNNoise for ABF_ALGO_RNNOISE, else bypass/OLA (or cdsp_select) */
#elif defined(BENCH_WITH_CDSP)
    abf_cdc_set_selector(cdsp_select);
#else
    abf_cdc_set_selector(basic_select);     /* the RNNoise image's build-twice twin */
#endif
}

/* The vector is declared .weak in startup_stm32f411xe.s, so defining it here
 * overrides Default_Handler. CubeMX did not generate one because the
 * USB_DEVICE middleware is disabled in the .ioc. */
void OTG_FS_IRQHandler(void)
{
    tud_int_handler(BOARD_TUD_RHPORT);
}
