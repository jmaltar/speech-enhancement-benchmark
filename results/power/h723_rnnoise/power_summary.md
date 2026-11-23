# power session: results/power/h723_rnnoise  (clk 550 MHz, hop 10 ms, budget 5,500,000 cycles)

| leg | n | span s | I mA | V | P mW | duty (board cycles / wall) | RTF | overruns | E mJ per audio-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| idle | 1614 | 57 | 181.51 ± 0.20 | 4.543 | 824.7 | 0.000 | 0.000 |  | nan |
| unpaced | 32846 | 1160 | 158.72 ± 4.72 | 4.571 | 725.4 | 0.578 | 0.323 | 0 | 405.2 |
| paced | 61921 | 2186 | 162.34 ± 2.82 | 4.569 | 741.7 | 0.307 | 0.322 | 0 | 781.0 |

P_idle = 824.7 mW, P_active (solved from the unpaced leg at duty 0.578) = 653.1 mW
paced, measured: 781.0 mJ per audio-second at RTF 0.322
corrected form P_active*RTF + P_idle*(1-RTF) = 769.4 mJ/s; the PDF's P_active*RTF alone = 210.6 mJ/s
