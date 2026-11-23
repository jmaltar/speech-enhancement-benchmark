/* INA226 power logger for the spare STM32F411 (WeAct Black Pill) -- the fixture of the power bench.
 *
 * The same board is also the USB breakout for the board under test: its USB-C data lines run straight through its
 * header pins A11/A12 to the USB-A breakout. So this firmware must leave USB alone: the OTG_FS peripheral is never
 * clocked and PA11/PA12 are put in analog mode (the most passive state a pin has) before anything else happens.
 * Readings leave over USART2 (PA2 = TX, PA3 = RX) to the debugger's serial port, never over this board's own USB.
 *
 *   I2C (software) PB6 = SCL, PB7 = SDA, ~90 kHz -> INA226 at 0x40 (A0 = A1 = GND)
 *   USART2 PA2 = TX, PA3 = RX, 115200 8N1        -> debugger RXD / TXD
 *   clock: HSI 16 MHz as after reset, no PLL     (no USB, so no 48 MHz requirement; a quiet board keeps the 5 V steady)
 *   TIM2: free-running 1 MHz, 32-bit             -> timestamps in microseconds (wraps every 71.6 min; the host unwraps)
 *   IWDG: ~1 s                                   -> a hung logger resets and prints its banner again, never goes silent
 *   PC13 LED (active low): on while the INA226 is missing or failing; one short blink per 32 readings otherwise
 *
 * Output, one line each, ASCII:
 *   # ...                                  banner and status (also in answer to '?')
 *   D,<seq>,<t_us>,<vshunt>,<vbus>          one reading: raw INA226 registers 0x01 (signed, 2.5 uV/LSB) and
 *                                          0x02 (1.25 mV/LSB), read as soon as a new averaged conversion is ready
 *   E,<code>,<where>                       an I2C or INA226 error; the logger retries and carries on
 * Commands (one line, from the host): '?' status, 'S' I2C bus scan, 'C xxxx' write a new INA226 configuration (hex),
 * 'R' reset. When the INA226 is missing at boot the scan runs by itself.
 *
 * VBS is wired to the supply side of the shunt (the module's IN- pin cannot take a second jumper), so the bus
 * register reads the supply-side voltage; the host reports the board's voltage as vbus - vshunt.
 */
#include <stdint.h>
#include "stm32f4xx.h"

#define FW_NAME            "ina226-logger"
#define FW_VERSION         "1"

#define INA_ADDR           0x40u
#define INA_REG_CONFIG     0x00u
#define INA_REG_SHUNT      0x01u
#define INA_REG_BUS        0x02u
#define INA_REG_MASK       0x06u
#define INA_REG_MFG        0xFEu
#define INA_REG_DIE        0xFFu
#define INA_MASK_CVRF      0x0008u
#define INA_MFG_TI         0x5449u
/* AVG = 16, VBUSCT = VSHCT = 1.1 ms, shunt and bus continuous: one averaged reading every 16 x 2.2 = 35.2 ms */
#define INA_CONFIG_DEFAULT 0x4527u

#define I2C_TIMEOUT_US     3000u
#define POLL_PERIOD_US     2000u
#define RETRY_PERIOD_US    1000000u

enum { I2C_OK = 0, I2C_ERR_BUSY, I2C_ERR_NACK, I2C_ERR_BUS, I2C_ERR_TIMEOUT, INA_ERR_ID, INA_ERR_CFG };
static const char *const err_name[] = {"ok", "busy", "nack", "bus", "timeout", "id", "cfg"};

static uint16_t g_config = INA_CONFIG_DEFAULT;
static uint16_t g_mfg, g_die;
static int g_ina_ok;
static uint32_t g_seq;
static uint32_t g_reset_flags;

/* ------------------------------------------------------------------ time */
static inline uint32_t now_us(void) { return TIM2->CNT; }
static void delay_us(uint32_t us) { uint32_t t0 = now_us(); while ((uint32_t)(now_us() - t0) < us) { } }

/* ------------------------------------------------------------------ UART */
/* The USART holds one received byte. Commands can arrive while a long reply is being sent, so received bytes are
 * moved into a ring buffer from inside the transmit wait as well as from the main loop; nothing is lost to overrun. */
static volatile char g_rx[64];
static volatile uint8_t g_rx_head, g_rx_tail;
static void rx_pump(void)
{
    while (USART2->SR & (USART_SR_RXNE | USART_SR_ORE)) {
        char c = (char)USART2->DR;                       /* reading DR also clears ORE */
        uint8_t next = (uint8_t)((g_rx_head + 1u) % sizeof g_rx);
        if (next != g_rx_tail) { g_rx[g_rx_head] = c; g_rx_head = next; }
    }
}
static void uart_putc(char c)
{
    while (!(USART2->SR & USART_SR_TXE)) rx_pump();
    USART2->DR = (uint8_t)c;
}
static void uart_puts(const char *s) { while (*s) uart_putc(*s++); }
static void uart_putu(uint32_t v)
{
    char b[11]; int n = 0;
    do { b[n++] = (char)('0' + v % 10u); v /= 10u; } while (v);
    while (n) uart_putc(b[--n]);
}
static void uart_puti(int32_t v)
{
    if (v < 0) { uart_putc('-'); uart_putu((uint32_t)(-(v + 1)) + 1u); } else uart_putu((uint32_t)v);
}
static void uart_puthex(uint32_t v, int digits)
{
    static const char hx[] = "0123456789ABCDEF";
    for (int i = digits - 1; i >= 0; --i) uart_putc(hx[(v >> (4 * i)) & 0xFu]);
}

/* ------------------------------------------------------------------ I2C, bit-banged on PB6 (SCL) / PB7 (SDA)
 * Software I2C, about 90 kHz: open-drain GPIO, a line is released (pulled up) or driven low, never driven high.
 * A software master has no hidden peripheral state and every step is visible, which is what a measurement logger
 * wants. (It replaced the I2C1 peripheral during bring-up, when reads failed; the cause turned out to be the module's
 * ground wire not being connected, not the peripheral.) Clock stretching is honoured with a bounded wait. */
#define SCL_PIN 6u
#define SDA_PIN 7u
#define I2C_HALF_US 5u
static uint8_t g_step;                                   /* which step failed last, for the error line */

static inline void scl_rel(void) { GPIOB->BSRR = 1u << SCL_PIN; }
static inline void scl_low(void) { GPIOB->BSRR = 1u << (SCL_PIN + 16u); }
static inline void sda_rel(void) { GPIOB->BSRR = 1u << SDA_PIN; }
static inline void sda_low(void) { GPIOB->BSRR = 1u << (SDA_PIN + 16u); }
static inline int scl_in(void) { return (GPIOB->IDR >> SCL_PIN) & 1u; }
static inline int sda_in(void) { return (GPIOB->IDR >> SDA_PIN) & 1u; }

static void i2c_pins_init(void)
{
    GPIOB->BSRR = (1u << SCL_PIN) | (1u << SDA_PIN);                       /* released before they become outputs */
    GPIOB->OTYPER |= (1u << SCL_PIN) | (1u << SDA_PIN);                   /* open drain */
    GPIOB->OSPEEDR = (GPIOB->OSPEEDR & ~((3u << 12) | (3u << 14))) | (1u << 12) | (1u << 14);
    GPIOB->PUPDR = (GPIOB->PUPDR & ~((3u << 12) | (3u << 14))) | (1u << 12) | (1u << 14);   /* + module pull-ups */
    GPIOB->MODER = (GPIOB->MODER & ~((3u << 12) | (3u << 14))) | (1u << 12) | (1u << 14);   /* outputs */
}

static int scl_high_wait(void)                           /* release SCL and wait for it: a slave may stretch */
{
    scl_rel();
    uint32_t t0 = now_us();
    while (!scl_in()) if ((uint32_t)(now_us() - t0) > I2C_TIMEOUT_US) return I2C_ERR_TIMEOUT;
    return I2C_OK;
}

static void i2c_stop(void)
{
    sda_low(); delay_us(I2C_HALF_US);
    (void)scl_high_wait(); delay_us(I2C_HALF_US);
    sda_rel(); delay_us(I2C_HALF_US);
}

/* A slave left holding SDA low (power cycle mid-byte) is released by clocking SCL until SDA is high, then a STOP. */
static void i2c_recover(void)
{
    sda_rel();
    for (int i = 0; i < 9 && !sda_in(); ++i) {
        scl_low(); delay_us(I2C_HALF_US);
        scl_rel(); delay_us(I2C_HALF_US);
    }
    i2c_stop();
}

static int i2c_start_cond(void)                          /* START, or repeated START */
{
    sda_rel(); delay_us(I2C_HALF_US);
    if (scl_high_wait()) return I2C_ERR_TIMEOUT;
    delay_us(I2C_HALF_US);
    if (!sda_in()) { i2c_recover(); if (!sda_in()) return I2C_ERR_BUSY; }
    sda_low(); delay_us(I2C_HALF_US);
    scl_low(); delay_us(I2C_HALF_US);
    return I2C_OK;
}

static int i2c_write_byte(uint8_t b)                     /* returns I2C_OK on ACK, I2C_ERR_NACK otherwise */
{
    for (int i = 7; i >= 0; --i) {
        if ((b >> i) & 1u) sda_rel(); else sda_low();
        delay_us(I2C_HALF_US);
        if (scl_high_wait()) return I2C_ERR_TIMEOUT;
        delay_us(I2C_HALF_US);
        scl_low();
    }
    sda_rel(); delay_us(I2C_HALF_US);                     /* the slave drives the ACK bit */
    if (scl_high_wait()) return I2C_ERR_TIMEOUT;
    int ack = !sda_in();
    delay_us(I2C_HALF_US);
    scl_low(); delay_us(I2C_HALF_US);
    return ack ? I2C_OK : I2C_ERR_NACK;
}

static int i2c_read_byte(int ack, uint8_t *out)
{
    uint8_t b = 0;
    sda_rel();
    for (int i = 0; i < 8; ++i) {
        delay_us(I2C_HALF_US);
        if (scl_high_wait()) return I2C_ERR_TIMEOUT;
        b = (uint8_t)((b << 1) | sda_in());
        delay_us(I2C_HALF_US);
        scl_low();
    }
    if (ack) sda_low(); else sda_rel();
    delay_us(I2C_HALF_US);
    if (scl_high_wait()) return I2C_ERR_TIMEOUT;
    delay_us(I2C_HALF_US);
    scl_low(); sda_rel(); delay_us(I2C_HALF_US);
    *out = b;
    return I2C_OK;
}

static int i2c_fail(int e) { i2c_stop(); return e; }

static int ina_write(uint8_t reg, uint16_t val)
{
    int e;
    g_step = 1;  if ((e = i2c_start_cond())) return i2c_fail(e);
    g_step = 2;  if ((e = i2c_write_byte((uint8_t)(INA_ADDR << 1)))) return i2c_fail(e);
    g_step = 3;  if ((e = i2c_write_byte(reg))) return i2c_fail(e);
    g_step = 4;  if ((e = i2c_write_byte((uint8_t)(val >> 8)))) return i2c_fail(e);
    g_step = 5;  if ((e = i2c_write_byte((uint8_t)val))) return i2c_fail(e);
    i2c_stop();
    return I2C_OK;
}

static int ina_read(uint8_t reg, uint16_t *val)
{
    int e;
    uint8_t hi, lo;
    g_step = 11; if ((e = i2c_start_cond())) return i2c_fail(e);
    g_step = 12; if ((e = i2c_write_byte((uint8_t)(INA_ADDR << 1)))) return i2c_fail(e);
    g_step = 13; if ((e = i2c_write_byte(reg))) return i2c_fail(e);
    g_step = 14; if ((e = i2c_start_cond())) return i2c_fail(e);                  /* repeated START */
    g_step = 15; if ((e = i2c_write_byte((uint8_t)((INA_ADDR << 1) | 1u)))) return i2c_fail(e);
    g_step = 16; if ((e = i2c_read_byte(1, &hi))) return i2c_fail(e);
    g_step = 17; if ((e = i2c_read_byte(0, &lo))) return i2c_fail(e);
    i2c_stop();
    *val = (uint16_t)((hi << 8) | lo);
    return I2C_OK;
}

/* ------------------------------------------------------------------ INA226 */
static void report_err(int e, const char *where)
{
    uart_puts("E,"); uart_puts(err_name[e]); uart_putc(','); uart_puts(where);
    uart_puts("@"); uart_putu(g_step); uart_puts("\r\n");
}

static int ina_configure(uint16_t cfg)
{
    int e;
    uint16_t rb;
    if ((e = ina_write(INA_REG_CONFIG, cfg))) return e;
    if ((e = ina_read(INA_REG_CONFIG, &rb))) return e;
    if ((rb & 0x0FFFu) != (cfg & 0x0FFFu)) return INA_ERR_CFG;   /* read the configuration back, never trust it */
    return I2C_OK;
}

static int ina_init(void)
{
    int e;
    if ((e = ina_read(INA_REG_MFG, &g_mfg))) return e;
    if ((e = ina_read(INA_REG_DIE, &g_die))) return e;
    if (g_mfg != INA_MFG_TI) return INA_ERR_ID;
    if ((e = ina_write(INA_REG_CONFIG, 0x8000u))) return e;      /* reset to power-on defaults */
    delay_us(1000);
    return ina_configure(g_config);
}

/* Scan 0x08-0x77 for any device that acknowledges, and report the idle bus levels: names a wiring fault. */
static void i2c_scan(void)
{
    uart_puts("# i2c scan: scl="); uart_putc(scl_in() ? '1' : '0');
    uart_puts(" sda="); uart_putc(sda_in() ? '1' : '0');
    uart_puts(" ack:");
    int n = 0;
    for (uint8_t a = 0x08; a < 0x78; ++a) {
        IWDG->KR = 0xAAAAu;
        if (i2c_start_cond()) { uart_puts(" bus-stuck"); i2c_stop(); break; }
        int e = i2c_write_byte((uint8_t)(a << 1));
        i2c_stop();
        if (e == I2C_OK) { uart_puts(" 0x"); uart_puthex(a, 2); ++n; }
    }
    if (!n) uart_puts(" none (no device answers: check the module's VCC to 3V3, GND, SCL to B6, SDA to B7)");
    uart_puts("\r\n");
}

/* ------------------------------------------------------------------ status */
static void banner(void)
{
    const uint32_t *uid = (const uint32_t *)UID_BASE;
    uart_puts("# " FW_NAME " " FW_VERSION " uid=");
    uart_puthex(uid[0], 8); uart_puthex(uid[1], 8); uart_puthex(uid[2], 8);
    uart_puts(" clk=hsi16 i2c=sw90k uart=115200\r\n");
    /* the guards a reader of the log can check: USB clock off, A11/A12 analog, running on HSI */
    uint32_t moder = (GPIOA->MODER >> 22) & 0xFu;
    uart_puts("# usb_clock="); uart_puts((RCC->AHB2ENR & RCC_AHB2ENR_OTGFSEN) ? "ON(!)" : "off");
    uart_puts(" pa11_pa12="); uart_puts(moder == 0xFu ? "analog" : "NOT-analog(!)");
    uart_puts(" sysclk="); uart_puts(((RCC->CFGR & RCC_CFGR_SWS) == RCC_CFGR_SWS_HSI) ? "hsi" : "NOT-hsi(!)");
    uart_puts(" reset=0x"); uart_puthex(g_reset_flags >> 24, 2);
    uart_puts("\r\n# ina226 ");
    if (g_ina_ok) {
        uart_puts("mfg=0x"); uart_puthex(g_mfg, 4); uart_puts(" die=0x"); uart_puthex(g_die, 4);
        uart_puts(" cfg=0x"); uart_puthex(g_config, 4);
    } else {
        uart_puts("not found (check SDA/SCL/VCC/GND; retrying every second)");
    }
    uart_puts("\r\n# fields D,seq,t_us,vshunt_raw,vbus_raw  lsb_shunt_nV=2500 lsb_bus_uV=1250"
              "  vbus is the supply side: v_board = vbus - vshunt\r\n");
}

/* ------------------------------------------------------------------ host commands */
static char g_cmd[24];
static unsigned g_cmd_len;

static int hexval(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static void run_command(void)
{
    g_cmd[g_cmd_len] = 0;
    if (g_cmd[0] == '?') {
        banner();
    } else if (g_cmd[0] == 'S') {
        i2c_scan();
    } else if (g_cmd[0] == 'R') {
        uart_puts("# reset\r\n");
        while (!(USART2->SR & USART_SR_TC)) { }
        NVIC_SystemReset();
    } else if (g_cmd[0] == 'C') {
        uint32_t v = 0; int digits = 0;
        for (unsigned i = 1; i < g_cmd_len; ++i) {
            int h = hexval(g_cmd[i]);
            if (h >= 0) { v = (v << 4) | (uint32_t)h; ++digits; }
            else if (g_cmd[i] != ' ' && !(g_cmd[i] == 'x' && v == 0)) { digits = -1; break; }
        }
        if (digits < 1 || digits > 4) { uart_puts("E,arg,C\r\n"); return; }
        int e = ina_configure((uint16_t)v);
        if (e) { report_err(e, "config"); return; }
        g_config = (uint16_t)v;
        uart_puts("# cfg=0x"); uart_puthex(g_config, 4); uart_puts("\r\n");
    } else if (g_cmd_len) {
        uart_puts("E,cmd,"); uart_puts(g_cmd); uart_puts("\r\n");
    }
}

static void poll_uart_rx(void)
{
    rx_pump();
    while (g_rx_tail != g_rx_head) {
        char c = g_rx[g_rx_tail];
        g_rx_tail = (uint8_t)((g_rx_tail + 1u) % sizeof g_rx);
        if (c == '\r' || c == '\n') { run_command(); g_cmd_len = 0; }
        else if (g_cmd_len < sizeof g_cmd - 1) g_cmd[g_cmd_len++] = c;
    }
}

/* ------------------------------------------------------------------ set-up */
static void board_init(void)
{
    g_reset_flags = RCC->CSR;
    RCC->CSR |= RCC_CSR_RMVF;

    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN | RCC_AHB1ENR_GPIOBEN | RCC_AHB1ENR_GPIOCEN;
    (void)RCC->AHB1ENR;

    /* First: PA11/PA12 (USB D-/D+) to analog, no pulls. They carry the board under test's USB traffic. */
    GPIOA->PUPDR &= ~((3u << 22) | (3u << 24));
    GPIOA->MODER |= (3u << 22) | (3u << 24);
    RCC->AHB2ENR &= ~RCC_AHB2ENR_OTGFSEN;                 /* already 0 after reset; stated, not assumed */

    /* PC13 LED, active low, off */
    GPIOC->BSRR = (1u << 13);
    GPIOC->MODER = (GPIOC->MODER & ~(3u << 26)) | (1u << 26);

    /* TIM2: 1 MHz free-running */
    RCC->APB1ENR |= RCC_APB1ENR_TIM2EN | RCC_APB1ENR_USART2EN;
    (void)RCC->APB1ENR;
    TIM2->PSC = 15u;
    TIM2->ARR = 0xFFFFFFFFu;
    TIM2->EGR = TIM_EGR_UG;
    TIM2->CR1 = TIM_CR1_CEN;

    /* USART2 on PA2 (TX) / PA3 (RX, pull-up so a disconnected debugger reads idle), AF7, 115200 8N1 */
    GPIOA->AFR[0] = (GPIOA->AFR[0] & ~((0xFu << 8) | (0xFu << 12))) | (7u << 8) | (7u << 12);
    GPIOA->PUPDR = (GPIOA->PUPDR & ~(3u << 6)) | (1u << 6);
    GPIOA->MODER = (GPIOA->MODER & ~((3u << 4) | (3u << 6))) | (2u << 4) | (2u << 6);
    USART2->BRR = 0x8Bu;                                  /* 16 MHz / 115200 = 8.68 -> 8 + 11/16 */
    USART2->CR1 = USART_CR1_UE | USART_CR1_TE | USART_CR1_RE;

    i2c_pins_init();
    i2c_recover();                                       /* a clean STOP on an idle bus is harmless */

    /* IWDG: LSI ~32 kHz / 32 = 1 kHz, reload 1000 -> ~1 s */
    IWDG->KR = 0xCCCCu;
    IWDG->KR = 0x5555u;
    IWDG->PR = 3u;
    IWDG->RLR = 1000u;
    while (IWDG->SR) { }
    IWDG->KR = 0xAAAAu;
}

static inline void led(int on) { GPIOC->BSRR = on ? (1u << (13 + 16)) : (1u << 13); }

/* ------------------------------------------------------------------ main */
int main(void)
{
    board_init();
    delay_us(20000);                                     /* let the INA226 finish its own power-up */

    int e = ina_init();
    g_ina_ok = (e == I2C_OK);
    banner();
    if (!g_ina_ok) { report_err(e, "init"); i2c_scan(); led(1); }

    uint32_t last_poll = now_us(), last_retry = now_us(), blink_until = 0;
    for (;;) {
        IWDG->KR = 0xAAAAu;
        poll_uart_rx();
        uint32_t t = now_us();

        if (!g_ina_ok) {
            if ((uint32_t)(t - last_retry) >= RETRY_PERIOD_US) {
                last_retry = t;
                e = ina_init();
                if (e == I2C_OK) { g_ina_ok = 1; led(0); banner(); }
                else report_err(e, "init");
            }
            continue;
        }
        if (blink_until && (int32_t)(t - blink_until) >= 0) { led(0); blink_until = 0; }
        if ((uint32_t)(t - last_poll) < POLL_PERIOD_US) continue;
        last_poll = t;

        uint16_t mask, shunt, bus;
        if ((e = ina_read(INA_REG_MASK, &mask))) { report_err(e, "mask"); g_ina_ok = 0; led(1); continue; }
        if (!(mask & INA_MASK_CVRF)) continue;            /* reading the mask register cleared the flag */
        uint32_t ts = now_us();
        if ((e = ina_read(INA_REG_SHUNT, &shunt))) { report_err(e, "shunt"); g_ina_ok = 0; led(1); continue; }
        if ((e = ina_read(INA_REG_BUS, &bus)))     { report_err(e, "bus");   g_ina_ok = 0; led(1); continue; }

        uart_puts("D,"); uart_putu(g_seq++); uart_putc(',');
        uart_putu(ts); uart_putc(',');
        uart_puti((int16_t)shunt); uart_putc(',');
        uart_putu(bus); uart_puts("\r\n");
        if ((g_seq & 31u) == 0) { led(1); blink_until = now_us() + 20000u; }
    }
}
