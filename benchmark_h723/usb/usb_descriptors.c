/**
 * usb_descriptors.c -- CDC ACM descriptors for the STM32H723 benchmark target.
 *
 * CDC rather than UAC is a locked decision: TinyUSB's UAC2 support is thinner
 * than its CDC support, and a custom framing over a bulk pipe gives explicit
 * control over packet boundaries, which is what the end-to-end latency and
 * jitter measurements need. The framing itself is task 7; this file only
 * describes the pipes it will run over.
 */
#include "tusb.h"
#include "main.h"   /* UID_BASE -- the 96-bit unique device ID */

/* 0xCafe is TinyUSB's example VID. It is not an allocated USB-IF vendor ID and
 * must not ship in a product, but this device never leaves the bench. The PID
 * is derived from the enabled class set the way TinyUSB's own examples do it,
 * so a future MSC or vendor-class build enumerates as a distinct device rather
 * than colliding in the host's driver cache. */
#define USB_VID   0xCafe
#define USB_BCD   0x0200   /* USB 2.0 full speed */
/* Bit 8 distinguishes the H723 from the F411 (0x4001), so both boards can be
 * plugged into the same host at once and stay separable in a bus trace. */
#define USB_PID   (0x4100 | (CFG_TUD_CDC << 0))

/* ---- device descriptor -------------------------------------------------
 * Class/subclass/protocol are MISC / common / IAD so the CDC interface
 * association descriptor below is honoured; without this a host binds the
 * first interface only and the data interface is left unclaimed.
 * ---------------------------------------------------------------------- */
static tusb_desc_device_t const desc_device = {
    .bLength            = sizeof(tusb_desc_device_t),
    .bDescriptorType    = TUSB_DESC_DEVICE,
    .bcdUSB             = USB_BCD,
    .bDeviceClass       = TUSB_CLASS_MISC,
    .bDeviceSubClass    = MISC_SUBCLASS_COMMON,
    .bDeviceProtocol    = MISC_PROTOCOL_IAD,
    .bMaxPacketSize0    = CFG_TUD_ENDPOINT0_SIZE,
    .idVendor           = USB_VID,
    .idProduct          = USB_PID,
    .bcdDevice          = 0x0100,
    .iManufacturer      = 0x01,
    .iProduct           = 0x02,
    .iSerialNumber      = 0x03,
    .bNumConfigurations = 0x01,
};

uint8_t const *tud_descriptor_device_cb(void)
{
    return (uint8_t const *)&desc_device;
}

/* ---- configuration descriptor ----------------------------------------- */
enum { ITF_NUM_CDC = 0, ITF_NUM_CDC_DATA, ITF_NUM_TOTAL };

#define CONFIG_TOTAL_LEN   (TUD_CONFIG_DESC_LEN + TUD_CDC_DESC_LEN)

#define EPNUM_CDC_NOTIF    0x81
#define EPNUM_CDC_OUT      0x02
#define EPNUM_CDC_IN       0x82

static uint8_t const desc_fs_configuration[] = {
    /* config number, interface count, string index, total length,
     * attributes, power in mA. Bus-powered at 100 mA: the board draws its own
     * supply from the USB-C connector, and the power-measurement plan depends
     * on that path staying the only one (the debugger's 3V3 is left
     * disconnected for the same reason). */
    TUD_CONFIG_DESCRIPTOR(1, ITF_NUM_TOTAL, 0, CONFIG_TOTAL_LEN, 0x00, 100),

    /* interface number, string index, notification EP, notification size,
     * data out EP, data in EP, data EP size */
    TUD_CDC_DESCRIPTOR(ITF_NUM_CDC, 4, EPNUM_CDC_NOTIF, 8,
                       EPNUM_CDC_OUT, EPNUM_CDC_IN, CFG_TUD_CDC_EP_BUFSIZE),
};

uint8_t const *tud_descriptor_configuration_cb(uint8_t index)
{
    (void)index;   /* one configuration only */
    return desc_fs_configuration;
}

/* ---- string descriptors ------------------------------------------------ */
enum { STRID_LANGID = 0, STRID_MANUFACTURER, STRID_PRODUCT, STRID_SERIAL, STRID_CDC };

static char const *string_desc_arr[] = {
    [STRID_LANGID]       = (const char[]){0x09, 0x04},   /* 0x0409 en-US */
    [STRID_MANUFACTURER] = "Benchmark",
    [STRID_PRODUCT]      = "STM32H723 Audio Bench",
    [STRID_SERIAL]       = NULL,                         /* filled from UID */
    [STRID_CDC]          = "Bench CDC",
};

static uint16_t _desc_str[32 + 1];

/* The 96-bit unique device ID becomes the serial number, so several boards on
 * one host enumerate distinctly and a captured trace can be attributed to a
 * specific board. Matters once the H723 is on the same bench. */
static size_t board_serial_hex(char *buf, size_t cap)
{
    static const char hex[] = "0123456789ABCDEF";
    const uint32_t *uid = (const uint32_t *)UID_BASE;
    size_t n = 0;
    for (int w = 0; w < 3 && n + 8 < cap; ++w) {
        for (int nib = 7; nib >= 0; --nib) {
            buf[n++] = hex[(uid[w] >> (nib * 4)) & 0xFu];
        }
    }
    buf[n] = '\0';
    return n;
}

uint16_t const *tud_descriptor_string_cb(uint8_t index, uint16_t langid)
{
    (void)langid;
    size_t chr_count = 0;

    if (index == STRID_LANGID) {
        memcpy(&_desc_str[1], string_desc_arr[STRID_LANGID], 2);
        chr_count = 1;
    } else if (index == STRID_SERIAL) {
        char serial[25];
        chr_count = board_serial_hex(serial, sizeof(serial));
        for (size_t i = 0; i < chr_count; ++i) _desc_str[1 + i] = (uint16_t)serial[i];
    } else {
        if (index >= TU_ARRAY_SIZE(string_desc_arr)) return NULL;
        const char *str = string_desc_arr[index];
        if (str == NULL) return NULL;

        chr_count = strlen(str);
        const size_t max_count = TU_ARRAY_SIZE(_desc_str) - 1;
        if (chr_count > max_count) chr_count = max_count;
        for (size_t i = 0; i < chr_count; ++i) _desc_str[1 + i] = (uint16_t)str[i];
    }

    /* first word is the length (bytes, header included) and the descriptor type */
    _desc_str[0] = (uint16_t)((TUSB_DESC_STRING << 8) | (2 * chr_count + 2));
    return _desc_str;
}
