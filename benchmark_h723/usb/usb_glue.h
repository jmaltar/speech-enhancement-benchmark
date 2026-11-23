#ifndef USB_GLUE_H
#define USB_GLUE_H
/* Brings up USB1_OTG_HS through the internal FS PHY (CRS source override,
 * kernel clock, transceiver supply, PA11/PA12, NVIC) and starts TinyUSB. Call
 * once, after the clock tree is configured. */
void usb_glue_init(void);

/* Computes the image CRC, fills the HELLO identity and initialises the ABF1
 * CDC binding. Call once, before usb_glue_init(). The data path itself is
 * abf_cdc_service(), called from the main loop beside tud_task(); the
 * placeholder echo is gone. */
void usb_glue_abf_init(void);
#endif
