# VBD ABBA benchmark: Isaac-Lift-Soft-Franka

- Baseline median: 24,610.636 FPS (8 fresh processes)
- Candidate median: 32,949.169 FPS (8 fresh processes)
- Paired median speedup: 1.3392x (+33.92%)
- 95.0% paired process-bootstrap CI: [1.3325x, 1.3654x]
- Ratio of variant medians: 1.3388x
- Outlier policy: all completed processes included; modified-z flags are diagnostics only.

| Pair | Baseline FPS | Candidate FPS | Speedup |
|---|---:|---:|---:|
| block_01_pair_1 | 24,092.760 | 32,910.857 | 1.3660x |
| block_01_pair_2 | 24,384.402 | 32,972.470 | 1.3522x |
| block_02_pair_1 | 24,923.618 | 33,143.865 | 1.3298x |
| block_02_pair_2 | 24,544.321 | 32,925.868 | 1.3415x |
| block_03_pair_1 | 24,792.201 | 33,102.514 | 1.3352x |
| block_03_pair_2 | 24,628.649 | 32,924.196 | 1.3368x |
| block_04_pair_1 | 24,785.300 | 33,841.371 | 1.3654x |
| block_04_pair_2 | 24,592.624 | 32,768.910 | 1.3325x |
