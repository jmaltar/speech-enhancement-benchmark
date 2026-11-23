#ifndef USB_GLUE_H
#define USB_GLUE_H
/* Brings up the OTG_FS peripheral (clock, PA11/PA12, NVIC) and starts
 * TinyUSB. Call once, after the clock tree is configured. */
void usb_glue_init(void);

/* Computes the image CRC, fills the HELLO identity and initialises the ABF1
 * CDC binding with this build's processor. Call once, before usb_glue_init(),
 * so the binding is ready for the first service call after enumeration. The
 * data path itself is abf_cdc_service(), called from the main loop beside
 * tud_task(); there is no usb_glue_service() any more. */
void usb_glue_abf_init(void);
#endif
