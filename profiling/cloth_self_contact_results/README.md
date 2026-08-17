# Cloth self-contact sanitized evidence

- `abba_runs.csv` contains every measured process from the final eight-block
  ABBA/BAAB comparison. The `result_path` values identify the original local
  evidence files; SHA-256 values are retained for integrity checks.
- `trace_components.csv` contains the structurally classified CUDA-kernel
  totals from the final baseline and candidate Nsight captures.
- `abba_analysis.json` is the strict ABBA analysis, including source/tool
  fingerprints, all run diagnostics, block ratios, and the bootstrap interval.
- `trace_analysis.json` is the sanitized trace analysis. It retains only an
  explicit environment allowlist and the selected kernel aggregates.

The secret-bearing raw `.nsys-rep` and SQLite files are intentionally excluded.
See `../CLOTH_SELF_CONTACT_REPORT.md` for the protocol, aggregate results, and
profiling caveats.
