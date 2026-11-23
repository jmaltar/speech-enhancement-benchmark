/* Minimal start-up for the logger: copy .data, zero .bss, run main. No libc, no FPU use, no interrupts enabled. */
#include <stdint.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
int main(void);

void Reset_Handler(void)
{
    uint32_t *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;
    for (dst = &_sbss; dst < &_ebss; ) *dst++ = 0;
    (void)main();
    for (;;) { }
}

/* Any fault parks the core; the independent watchdog then resets it within a second. */
void Default_Handler(void) { for (;;) { } }

__attribute__((section(".isr_vector"), used))
static void (*const vectors[16])(void) = {
    (void (*)(void))(&_estack),
    Reset_Handler,
    Default_Handler,   /* NMI */
    Default_Handler,   /* HardFault */
    Default_Handler,   /* MemManage */
    Default_Handler,   /* BusFault */
    Default_Handler,   /* UsageFault */
    0, 0, 0, 0,
    Default_Handler,   /* SVCall */
    Default_Handler,   /* DebugMonitor */
    0,
    Default_Handler,   /* PendSV */
    Default_Handler,   /* SysTick */
};
