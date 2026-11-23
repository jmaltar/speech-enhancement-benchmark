/* rnnoise_place.c -- see rnnoise_place.h. Plain word loops, no memcpy: libc may
 * itself be in the ITCM section being filled. */
#include "rnnoise_place.h"
extern uint32_t _sitcm, _eitcm, _siitcm;
extern uint32_t _saxirodata, _eaxirodata, _siaxirodata;
volatile uint32_t g_place_itcm_addr, g_place_itcm_bytes, g_place_axi_addr, g_place_axi_bytes, g_place_verified;

static uint32_t copy_verify(uint32_t *dst, const uint32_t *end, const uint32_t *src)
{
    uint32_t *d = dst; const uint32_t *s = src;
    while (d < end) *d++ = *s++;
    __asm volatile ("dsb; isb" ::: "memory");
    d = dst; s = src;
    while (d < end) { if (*d++ != *s++) return 0u; }
    return 1u;
}

void rnnoise_place_init(void)
{
    uint32_t ok = 1u;
    g_place_itcm_bytes = (uint32_t)((uintptr_t)&_eitcm - (uintptr_t)&_sitcm);
    g_place_axi_bytes  = (uint32_t)((uintptr_t)&_eaxirodata - (uintptr_t)&_saxirodata);
    g_place_itcm_addr  = g_place_itcm_bytes ? (uint32_t)(uintptr_t)&_sitcm : 0u;
    g_place_axi_addr   = g_place_axi_bytes  ? (uint32_t)(uintptr_t)&_saxirodata : 0u;
    if (g_place_itcm_bytes) ok &= copy_verify(&_sitcm, &_eitcm, &_siitcm);
    if (g_place_axi_bytes)  ok &= copy_verify(&_saxirodata, &_eaxirodata, &_siaxirodata);
    g_place_verified = ok;
}
