# power session: /dev/shm/results_power_rnnoise  (clk 1000 MHz, hop 10 ms, budget 10,000,000 cycles)

| leg | n | span s | I mA | V | P mW | duty (board cycles / wall) | RTF | overruns | E mJ per audio-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| idle | 57 | 56 | 379.69 ± 12.58 | 5.136 | 1949.9 | 0.000 | 0.000 |  | nan |
| unpaced | 63 | 62 | 566.84 ± 69.64 | 5.140 | 2913.4 | 0.570 | 0.017 | 0 | 87.0 |
| paced | 2104 | 2103 | 396.15 ± 35.27 | 5.133 | 2033.4 | 0.017 | 0.017 | 0 | 2059.8 |

P_idle = 1949.9 mW, P_active (solved from the unpaced leg at duty 0.570) = 3641.2 mW
paced, measured: 2059.8 mJ per audio-second at RTF 0.017
corrected form P_active*RTF + P_idle*(1-RTF) = 1978.7 mJ/s; the PDF's P_active*RTF alone = 61.9 mJ/s
