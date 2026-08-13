# VBD ABBA benchmark: Isaac-Lift-Soft-Franka

- Baseline median: 24,576.049 FPS (8 fresh processes)
- Candidate median: 29,930.147 FPS (8 fresh processes)
- Paired median speedup: 1.2220x (+22.20%)
- 95.0% paired process-bootstrap CI: [1.1693x, 1.2435x]
- Ratio of variant medians: 1.2179x
- Outlier policy: all completed processes included; modified-z flags are diagnostics only.

| Pair | Baseline FPS | Candidate FPS | Speedup |
|---|---:|---:|---:|
| block_01_pair_1 | 24,289.853 | 29,951.551 | 1.2331x |
| block_01_pair_2 | 24,645.821 | 29,985.170 | 1.2166x |
| block_02_pair_1 | 24,368.451 | 29,908.744 | 1.2274x |
| block_02_pair_2 | 25,128.707 | 28,776.671 | 1.1452x |
| block_03_pair_1 | 25,129.101 | 31,538.598 | 1.2551x |
| block_03_pair_2 | 24,989.965 | 31,074.886 | 1.2435x |
| block_04_pair_1 | 24,506.277 | 29,293.729 | 1.1954x |
| block_04_pair_2 | 23,248.246 | 27,183.430 | 1.1693x |
