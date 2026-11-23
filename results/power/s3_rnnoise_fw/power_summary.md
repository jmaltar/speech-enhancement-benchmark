# power session: results/power/s3_rnnoise_fw  (clk 240 MHz, hop 10 ms, budget 2,400,000 cycles)

| leg | n | span s | I mA | V | P mW | duty (board cycles / wall) | RTF | overruns | E mJ per audio-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| idle_boot | 482 | 17 | 45.41 ± 0.04 | 4.702 | 213.5 | 0.000 | 0.000 |  | nan |
| idle | 1615 | 57 | 45.64 ± 0.07 | 4.702 | 214.6 | 0.000 | 0.000 |  | nan |
| unpaced | 68582 | 2421 | 76.65 ± 5.71 | 4.663 | 357.4 | 0.771 | 0.898 | 0 | 416.8 |
| paced | 68557 | 2420 | 76.68 ± 5.56 | 4.663 | 357.6 | 0.771 | 0.898 | 0 | 416.8 |

P_idle = 214.6 mW, P_active (solved from the unpaced leg at duty 0.771) = 399.9 mW
paced, measured: 416.8 mJ per audio-second at RTF 0.898
corrected form P_active*RTF + P_idle*(1-RTF) = 381.0 mJ/s; the PDF's P_active*RTF alone = 359.2 mJ/s
