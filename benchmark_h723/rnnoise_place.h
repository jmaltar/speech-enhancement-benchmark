/* rnnoise_place.h -- boot-time copies for the two linker-decided placement
 * variables of the H723 tier (bench_place.ld.in): code into ITCM, weights into
 * AXI SRAM. Call rnnoise_place_init() as the FIRST statement of main(): before
 * the caches are enabled (so the AXI copy lands in RAM, not in dirty lines) and
 * before anything that may have been moved is called. The witnesses are read
 * over SWD after a run (read_rnnoise_stamps.sh prints them) and are the proof
 * that the copy happened and verified; the ELF's symbol table is the proof of
 * WHAT was moved. */
#ifndef RNNOISE_PLACE_H
#define RNNOISE_PLACE_H
#include <stdint.h>
extern volatile uint32_t g_place_itcm_addr;    /* _sitcm (0x0000xxxx) or 0 if empty */
extern volatile uint32_t g_place_itcm_bytes;
extern volatile uint32_t g_place_axi_addr;     /* _saxirodata (0x2400xxxx) or 0    */
extern volatile uint32_t g_place_axi_bytes;
extern volatile uint32_t g_place_verified;     /* 1 = both copies memcmp-identical to their flash images */
void rnnoise_place_init(void);
#endif
