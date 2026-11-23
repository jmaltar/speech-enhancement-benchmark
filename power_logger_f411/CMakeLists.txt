# INA226 power logger for the spare STM32F411 (power-bench fixture). Plain CMake, arm-none-eabi-gcc.
#   cmake -S power_logger_f411 -B power_logger_f411/build -G Ninja && cmake --build power_logger_f411/build
# CMSIS headers are taken from the F411 benchmark tree (the pinned STM32Cube FW_F4 copy), not duplicated here.
cmake_minimum_required(VERSION 3.20)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

project(ina226_logger C)

set(CMSIS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../benchmark_f411/Drivers/CMSIS" CACHE PATH "CMSIS (STM32Cube FW_F4)")
if(NOT EXISTS "${CMSIS_ROOT}/Device/ST/STM32F4xx/Include/stm32f411xe.h")
  message(FATAL_ERROR "CMSIS device headers not found under ${CMSIS_ROOT}")
endif()

set(MCU_FLAGS -mcpu=cortex-m4 -mthumb -mfloat-abi=soft)
add_executable(ina226_logger main.c startup.c)
set_target_properties(ina226_logger PROPERTIES SUFFIX ".elf")
target_include_directories(ina226_logger PRIVATE
  "${CMSIS_ROOT}/Include" "${CMSIS_ROOT}/Core/Include" "${CMSIS_ROOT}/Device/ST/STM32F4xx/Include")
target_compile_definitions(ina226_logger PRIVATE STM32F411xE)
target_compile_options(ina226_logger PRIVATE ${MCU_FLAGS} -O2 -g -std=c11 -ffreestanding -fno-common
  -ffunction-sections -fdata-sections -Wall -Wextra -Werror)
target_link_options(ina226_logger PRIVATE ${MCU_FLAGS} -nostdlib -nostartfiles -Wl,--gc-sections
  -T "${CMAKE_CURRENT_SOURCE_DIR}/logger.ld" -Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/ina226_logger.map)
target_link_libraries(ina226_logger PRIVATE gcc)

add_custom_command(TARGET ina226_logger POST_BUILD
  COMMAND arm-none-eabi-objcopy -O binary $<TARGET_FILE:ina226_logger> ${CMAKE_CURRENT_BINARY_DIR}/ina226_logger.bin
  COMMAND arm-none-eabi-size $<TARGET_FILE:ina226_logger>
  COMMENT "ina226_logger.bin")
