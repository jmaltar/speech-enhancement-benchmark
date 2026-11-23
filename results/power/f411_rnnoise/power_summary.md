# power session: results/power/f411_rnnoise  (clk 96 MHz, hop 10 ms, budget 960,000 cycles)

| leg | n | span s | I mA | V | P mW | duty (board cycles / wall) | RTF | overruns | E mJ per audio-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| idle_boot | 482 | 17 | 22.06 ± 0.26 | 4.733 | 104.4 | 0.000 | 0.000 |  | nan |
| idle | 1615 | 57 | 27.57 ± 0.29 | 4.725 | 130.3 | 0.000 | 0.000 |  | nan |
| unpaced | 202960 | 7165 | 31.23 ± 0.67 | 4.725 | 147.6 | 0.920 | 3.167 | 208429 | 509.4 |
| paced | 202967 | 7166 | 31.11 ± 0.65 | 4.722 | 146.9 | 0.919 | 3.167 | 208429 | 507.1 |

P_idle = 130.3 mW, P_active (solved from the unpaced leg at duty 0.920) = 149.1 mW
paced, measured: 507.1 mJ per audio-second at RTF 3.167
corrected form P_active*RTF + P_idle*(1-RTF) = 189.9 mJ/s; the PDF's P_active*RTF alone = 472.3 mJ/s
