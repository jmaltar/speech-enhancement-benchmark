/**
 * tusb_config.h -- TinyUSB configuration for the STM32F411 benchmark target.
 *
 * This file is a recorded output of task 3 as much as a build input: the CDC
 * FIFO sizes below are the single largest contributor to the stack's RAM
 * footprint, so the "SRAM remaining after the USB stack" figure in the budget
 * table is only meaningful alongside them.
 */
#ifndef TUSB_CONFIG_H
#define TUSB_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

/* ---- target ---------------------------------------------------------- */
#define CFG_TUSB_MCU              OPT_MCU_STM32F4
#define CFG_TUSB_OS               OPT_OS_NONE
#define CFG_TUSB_DEBUG            0

#define BOARD_TUD_RHPORT          0
#define BOARD_TUD_MAX_SPEED       OPT_MODE_FULL_SPEED

#define CFG_TUD_ENABLED           1
#define CFG_TUD_MAX_SPEED         BOARD_TUD_MAX_SPEED

/* The WeAct Black Pill does not wire PA9 to VBUS, so hardware VBUS detection
 * would leave the device permanently "disconnected" and it would never
 * enumerate. Same reason VBUS sensing is off in the CubeMX .ioc. This is also
 * why tud_umount_cb() cannot be relied on here -- without a VBUS pin TinyUSB
 * has no way to see the cable being pulled. */
#define CFG_TUD_VBUS_DETECT_HW    0

/* ---- memory ---------------------------------------------------------- */
#define CFG_TUSB_MEM_SECTION
#define CFG_TUSB_MEM_ALIGN        __attribute__ ((aligned(4)))

/* ---- classes --------------------------------------------------------- */
#define CFG_TUD_ENDPOINT0_SIZE    64
#define CFG_TUD_CDC               1
#define CFG_TUD_MSC               0
#define CFG_TUD_HID               0
#define CFG_TUD_MIDI              0
#define CFG_TUD_AUDIO             0
#define CFG_TUD_VENDOR            0

/* One 32 ms frame of the locked configuration is 512 samples of int16 =
 * 1024 bytes, so each FIFO holds exactly one frame. That is the smallest size
 * that lets task 7 move whole frames without splitting them across transfers,
 * and it makes the RAM line item traceable to an audio quantity rather than to
 * a round number. Revisit together with the framing design, not on its own. */
#define CFG_TUD_CDC_RX_BUFSIZE    1024
#define CFG_TUD_CDC_TX_BUFSIZE    1024
#define CFG_TUD_CDC_EP_BUFSIZE    64      /* full-speed bulk max packet */

#ifdef __cplusplus
}
#endif

#endif /* TUSB_CONFIG_H */
