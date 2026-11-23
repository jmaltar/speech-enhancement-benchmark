/*
 * dtln_fc_pie.S -- int16 x int8 dot product on the ESP32-S3 PIE, 2026-10-22.
 *
 *   int64_t dtln_pie_dot_s16s8(const int16_t *x, const int8_t *w, int n);
 *   a2 = x (16-byte aligned), a3 = w (16-byte aligned), a4 = n (multiple of 16)
 *   returns the 40-bit ACCX sum sign-extended to int64 in a2 (low) : a3 (high)
 *
 * Widening idiom as esp-nn's own s8 kernel uses it (esp_nn_fully_connected_
 * s8_esp32s3.S): EE.VCMP.LT.S8 against a zero register yields 0xFF where the
 * byte is negative, and EE.VZIP.8 interleaves the bytes with those masks, so
 * a 128-bit load of 16 int8 weights becomes two registers of 8 sign-extended
 * int16 lanes. EE.VMULAS.S16.ACCX then multiplies 8 int16 pairs and adds all
 * eight products into the scalar 40-bit ACCX in one instruction. 32 MACs per
 * loop iteration in 14 instructions: ~0.44 cycles per MAC if every one issues
 * in a cycle, against ~24 for the reference C++ (benchmark_s3/RESULTS.md).
 *
 * UNVERIFIED ON HARDWARE until dtln_fc_pie_selfcheck() has run on the board;
 * the processor refuses setup (g_dtln.setup_rc = 40 + case) if it disagrees
 * with the C dot product, so a wrong kernel cannot produce a measurement.
 */
    .text
    .align  4
    .literal_position
    .global dtln_pie_dot_s16s8
    .type   dtln_pie_dot_s16s8, @function
dtln_pie_dot_s16s8:
    entry   a1, 32
    ee.zero.accx
    ee.zero.q   q7                      /* zero lanes for the sign compare   */
    srai    a5, a4, 5                   /* 32-MAC iterations                 */
    loopgtz a5, .Lpie_loop_end
    ee.vld.128.ip   q0, a3, 16          /* wA[0..15]                         */
    ee.vld.128.ip   q4, a3, 16          /* wB[0..15]                         */
    ee.vld.128.ip   q1, a2, 16          /* xA[0..7]                          */
    ee.vld.128.ip   q2, a2, 16          /* xA[8..15]                         */
    ee.vcmp.lt.s8   q3, q0, q7
    ee.vzip.8       q0, q3              /* q0 = wA[0..7] s16, q3 = wA[8..15] */
    ee.vcmp.lt.s8   q5, q4, q7
    ee.vmulas.s16.accx q0, q1
    ee.vld.128.ip   q6, a2, 16          /* xB[0..7]                          */
    ee.vzip.8       q4, q5              /* q4 = wB[0..7] s16, q5 = wB[8..15] */
    ee.vmulas.s16.accx q3, q2
    ee.vld.128.ip   q1, a2, 16          /* xB[8..15] (q1 is free again)      */
    ee.vmulas.s16.accx q4, q6
    ee.vmulas.s16.accx q5, q1
.Lpie_loop_end:
    bbci    a4, 4, .Lpie_done           /* no 16-MAC remainder               */
    ee.vld.128.ip   q0, a3, 16
    ee.vld.128.ip   q1, a2, 16
    ee.vld.128.ip   q2, a2, 16
    ee.vcmp.lt.s8   q3, q0, q7
    ee.vzip.8       q0, q3
    ee.vmulas.s16.accx q0, q1
    ee.vmulas.s16.accx q3, q2
.Lpie_done:
    rur.accx_0  a2                      /* bits 31:0                         */
    rur.accx_1  a3                      /* bits 39:32                        */
    slli    a3, a3, 24                  /* sign-extend the 8-bit high part   */
    srai    a3, a3, 24
    retw.n
    .size   dtln_pie_dot_s16s8, . - dtln_pie_dot_s16s8
