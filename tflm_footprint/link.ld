/* Minimal generic Cortex-M map: only used to produce a linked image whose
   .text/.rodata can be measured. Sizes deliberately generous. */
MEMORY {
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 320K
}
ENTRY(Reset_Handler)
SECTIONS {
  .isr_vector : { KEEP(*(.isr_vector)) } > FLASH
  .text : { *(.text*) *(.glue_7*) KEEP(*(.init)) KEEP(*(.fini)) } > FLASH
  .rodata : { *(.rodata*) } > FLASH
  .ARM.extab : { *(.ARM.extab*) } > FLASH
  .ARM : { __exidx_start = .; *(.ARM.exidx*) __exidx_end = .; } > FLASH
  .preinit_array : { PROVIDE_HIDDEN(__preinit_array_start = .); KEEP(*(.preinit_array*)); PROVIDE_HIDDEN(__preinit_array_end = .); } > FLASH
  .init_array : { PROVIDE_HIDDEN(__init_array_start = .); KEEP(*(SORT(.init_array.*))); KEEP(*(.init_array*)); PROVIDE_HIDDEN(__init_array_end = .); } > FLASH
  .fini_array : { PROVIDE_HIDDEN(__fini_array_start = .); KEEP(*(SORT(.fini_array.*))); KEEP(*(.fini_array*)); PROVIDE_HIDDEN(__fini_array_end = .); } > FLASH
  _sidata = LOADADDR(.data);
  .data : { _sdata = .; *(.data*) _edata = .; } > RAM AT> FLASH
  .bss : { _sbss = .; *(.bss*) *(COMMON) _ebss = .; } > RAM
  ._user_heap_stack : { . = ALIGN(8); . = . + 0x400; . = ALIGN(8); } > RAM
  _estack = ORIGIN(RAM) + LENGTH(RAM);
  /DISCARD/ : { *(.comment) }
}
