/**
 * usb_glue.c -- board-side bring-up for TinyUSB on the WeAct STM32H723.
 *
 * TinyUSB's dwc2 port drives the OTG registers directly; it does not use ST's
 * HAL PCD driver, and it does not configure the peripheral's kernel clock,
 * supply, pins or interrupt. Those are this file's job.
 *
 * WHY THIS FILE IS BIGGER THAN THE F411's -- read before trimming it.
 *
 * On the F411, CubeMX's HAL_PCD_MspInit() only did two things (GPIO + the
 * peripheral clock), so deleting the MX_USB_OTG_FS_PCD_Init() call cost
 * nothing that was not trivially replaced. On the H7 the generated MspInit
 * ALSO selects the USB kernel clock and enables the transceiver's 3.3 V
 * supply, and CubeMX emits no GPIO setup at all. Delete the HAL call here
 * without replacing all of that and the result is a device that never
 * enumerates, with nothing visibly wrong anywhere in the code.
 *
 * The three things CubeMX stranded in Core/Src/usb_otg.c:
 *   1. HAL_RCCEx_PeriphCLKConfig(RCC_PERIPHCLK_USB -> HSI48)
 *   2. HAL_PWREx_EnableUSBVoltageDetector()
 *   3. __HAL_RCC_USB_OTG_HS_CLK_ENABLE()
 * ...plus one it never generated: the PA11/PA12 alternate-function setup.
 * ST's own H7 USB device examples do configure those pins, and the H7 GPIO
 * header documents GPIO_AF10_OTG1_FS as the mapping for STM32H72x/H73x.
 */
#include "usb_glue.h"
#include "main.h"
#include "tusb.h"
#include "abf_cdc.h"
#include "abf_proc.h"
#include "cdsp.h"
#ifdef BENCH_WITH_DTLN
#include "dtln_proc.h"
#endif
#ifdef BENCH_WITH_RNNOISE
#include "rnnoise_proc.h"
#endif

#if defined(BENCH_WITH_RNNOISE) && defined(BENCH_WITH_DTLN)
/* Both networks in one image: RNNoise's id to its selector, the rest to
 * DTLN's, which falls through to cdsp_select for bypass / OLA / classical. */
static const abf_proc_t *h723_select(const abf_config_t *cfg)
{
    if ((cfg->algo_id & ABF_ALGO_MASK) == ABF_ALGO_RNNOISE) return rnnoise_select(cfg);
    return dtln_select(cfg);
}
#endif

void usb_glue_init(void)
{
    RCC_PeriphCLKInitTypeDef pclk = {0};
    GPIO_InitTypeDef gpio = {0};

    /* 0. CRS sync source override -- DECIDED EMPIRICALLY 2026-09-11, not argued.
     *
     *    CubeMX generates RCC_CRS_SYNC_SOURCE_USB2 in SystemClock_Config() on a
     *    part whose only USB instance is USB1_OTG_HS. Nothing in the HAL catches
     *    it: the RCC_CRS_SYNC_SOURCE_* constants are not device-guarded and
     *    IS_RCC_CRS_SYNC_SOURCE() accepts all four on every H7.
     *
     *    Measured on hardware before this override existed: with the device
     *    enumerated and the host sending SOF, CRS->ISR read 0x00000000 -- SYNCOKF
     *    never set, but SYNCMISS and SYNCERR were clear too. The reason is in
     *    CFGR, not ISR: USB2 is SYNCSRC=0b11, a reserved encoding here, and the
     *    field does not latch it. CRS->CFGR read back 0x0022BB7F, i.e. SYNCSRC=0b00
     *    (external pin, wired to nothing), against a reset value of 0x2022BB7F
     *    (SYNCSRC=0b10, USB1). So the write executed and was silently rejected,
     *    and CRS sat waiting on a pin for a SOF that arrives on USB1.
     *
     *    Re-run here rather than edited into Core/Src/main.c so a CubeMX
     *    regeneration cannot quietly revert it. CRS trims HSI48 against host SOF,
     *    so it can only lock AFTER enumeration -- it is drift insurance across
     *    temperature, never a prerequisite for the first enumeration.
     */
    {
        RCC_CRSInitTypeDef crs = {0};
        crs.Prescaler             = RCC_CRS_SYNC_DIV1;
        crs.Source                = RCC_CRS_SYNC_SOURCE_USB1;
        crs.Polarity              = RCC_CRS_SYNC_POLARITY_RISING;
        crs.ReloadValue           = __HAL_RCC_CRS_RELOADVALUE_CALCULATE(48000000, 1000);
        crs.ErrorLimitValue       = 34;
        crs.HSI48CalibrationValue = 32;
        __HAL_RCC_CRS_CLK_ENABLE();
        HAL_RCCEx_CRSConfig(&crs);

        /* Same rule as the cache bits in bench_plat_init(): never trust the
         * #define that asked for it -- confirm the field actually took. This is
         * the exact failure the USB2 setting exhibited. */
        if (((CRS->CFGR & CRS_CFGR_SYNCSRC) != RCC_CRS_SYNC_SOURCE_USB1)) {
            Error_Handler();
        }
    }

    /* 1. USB kernel clock = HSI48 (trimmed by CRS against SOF; see
     *    SystemClock_Config). Re-asserted here rather than assumed, because
     *    this is the only place the USB stack's clock is now established. */
    pclk.PeriphClockSelection = RCC_PERIPHCLK_USB;
    pclk.UsbClockSelection    = RCC_USBCLKSOURCE_HSI48;
    if (HAL_RCCEx_PeriphCLKConfig(&pclk) != HAL_OK) {
        Error_Handler();
    }

    /* 2. Transceiver supply. Without this the PHY is unpowered and the bus
     *    never sees a pull-up, so the host does not even detect an insertion. */
    HAL_PWREx_EnableUSBVoltageDetector();

    /* 3. PA11 = OTG_HS_DM, PA12 = OTG_HS_DP, through the internal FS PHY.
     *    GPIO_AF10_OTG1_FS is the H72x/H73x mapping (stm32h7xx_hal_gpio_ex.h).
     *    VERY_HIGH speed is not optional: full-speed edges do not meet the eye
     *    diagram at lower drive settings. */
    __HAL_RCC_GPIOA_CLK_ENABLE();
    gpio.Pin       = GPIO_PIN_11 | GPIO_PIN_12;
    gpio.Mode      = GPIO_MODE_AF_PP;
    gpio.Pull      = GPIO_NOPULL;
    gpio.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;
    gpio.Alternate = GPIO_AF10_OTG1_FS;
    HAL_GPIO_Init(GPIOA, &gpio);

    /* 4. Peripheral clock. The ULPI clock is left alone: this part drives the
     *    internal PHY, and the generated MspInit does not touch ULPI either. */
    __HAL_RCC_USB_OTG_HS_CLK_ENABLE();

    /* Priority 5, matching the F411, so interrupt latency is not a silent
     * difference between the two platforms' jitter figures. */
    HAL_NVIC_SetPriority(OTG_HS_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(OTG_HS_IRQn);

    tusb_rhport_init_t init = {
        .role  = TUSB_ROLE_DEVICE,
        .speed = TUSB_SPEED_FULL,
    };
    tusb_init(BOARD_TUD_RHPORT, &init);
}

/* ---- identity, as on the F411 ---------------------------------------------
 * HELLO.image_crc32 is the CRC-32 of the flash image as it sits in flash,
 * vector table through the end of .data's load image -- the span the .bin
 * covers -- so the host's --check-bin is verify_image at the protocol level.
 * Flash the .bin, archive it, point every batch at the archive. */
extern uint32_t _sidata, _sdata, _edata;   /* from STM32H723xG_flash.ld */

static uint32_t image_crc32(void)
{
    const uint8_t *start = (const uint8_t *)FLASH_BASE;
    const uint8_t *end   = (const uint8_t *)&_sidata
                         + ((uintptr_t)&_edata - (uintptr_t)&_sdata);
    return abf_crc32_final(abf_crc32(ABF_CRC_INIT, start, (size_t)(end - start)));
}

#ifndef ABF_BUILD_ID
#  define ABF_BUILD_ID "unstamped"
#endif

void usb_glue_abf_init(void)
{
    abf_cdc_ident_t id = {
        .platform    = "STM32H723",
        .build_id    = ABF_BUILD_ID,
        .image_crc32 = image_crc32(),
        .cpu_hz      = SystemCoreClock,
    };
    abf_cdc_init(abf_proc_bypass(), &id);
#if defined(BENCH_WITH_RNNOISE) && defined(BENCH_WITH_DTLN)
    abf_cdc_set_selector(h723_select);   /* RNNoise, DTLN, else cdsp_select */
#elif defined(BENCH_WITH_RNNOISE)
    abf_cdc_set_selector(rnnoise_select); /* RNNoise for ABF_ALGO_RNNOISE, else cdsp_select */
#elif defined(BENCH_WITH_DTLN)
    abf_cdc_set_selector(dtln_select);   /* DTLN for ABF_ALGO_DTLN, else cdsp_select */
#else
    abf_cdc_set_selector(cdsp_select);   /* bypass / OLA / classical from CONFIG.algo_id */
#endif
}

/* The H723 has no OTG_FS: its single instance is USB1_OTG_HS, so the vector is
 * OTG_HS_IRQHandler, not the F411's OTG_FS_IRQHandler. rhport is still 0 --
 * TinyUSB aliases the FS base address onto the HS one in dwc2_stm32.h.
 * Declared .weak in startup_stm32h723xx.s, so this overrides Default_Handler. */
void OTG_HS_IRQHandler(void)
{
    tud_int_handler(BOARD_TUD_RHPORT);
}
