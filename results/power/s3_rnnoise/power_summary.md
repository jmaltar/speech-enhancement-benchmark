# power session: results/power/s3_rnnoise  (clk 240 MHz, hop 10 ms, budget 2,400,000 cycles)

| leg | n | span s | I mA | V | P mW | duty (board cycles / wall) | RTF | overruns | E mJ per audio-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| idle_boot | 482 | 17 | 44.98 ± 0.04 | 4.703 | 211.5 | 0.000 | 0.000 |  | nan |
| idle | 1614 | 57 | 45.16 ± 0.06 | 4.703 | 212.4 | 0.000 | 0.000 |  | nan |
| unpaced | 92293 | 3258 | 74.63 ± 4.71 | 4.666 | 348.2 | 0.832 | 1.305 | 208429 | 546.4 |
| paced | 92386 | 3261 | 73.46 ± 4.70 | 4.666 | 342.7 | 0.831 | 1.305 | 208429 | 538.4 |

P_idle = 212.4 mW, P_active (solved from the unpaced leg at duty 0.832) = 375.6 mW
paced, measured: 538.4 mJ per audio-second at RTF 1.305
corrected form P_active*RTF + P_idle*(1-RTF) = 425.3 mJ/s; the PDF's P_active*RTF alone = 490.1 mJ/s
