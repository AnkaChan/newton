# VBD ABBA benchmark: Isaac-Lift-Cloth-Franka

- Baseline median: 15,838.790 FPS (8 fresh processes)
- Candidate median: 25,223.074 FPS (8 fresh processes)
- Paired median speedup: 1.5948x (+59.48%)
- 95.0% paired process-bootstrap CI: [1.5582x, 1.6018x]
- Ratio of variant medians: 1.5925x
- Outlier policy: all completed processes included; modified-z flags are diagnostics only.

| Pair | Baseline FPS | Candidate FPS | Speedup |
|---|---:|---:|---:|
| block_01_pair_1 | 15,797.272 | 25,238.994 | 1.5977x |
| block_01_pair_2 | 15,815.617 | 24,799.811 | 1.5681x |
| block_02_pair_1 | 15,829.416 | 25,207.155 | 1.5924x |
| block_02_pair_2 | 15,968.258 | 25,578.207 | 1.6018x |
| block_03_pair_1 | 15,885.119 | 24,356.547 | 1.5333x |
| block_03_pair_2 | 15,811.998 | 25,454.255 | 1.6098x |
| block_04_pair_1 | 15,848.164 | 25,312.229 | 1.5972x |
| block_04_pair_2 | 16,009.317 | 24,945.418 | 1.5582x |
