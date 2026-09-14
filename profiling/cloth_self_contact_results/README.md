# Cloth self-contact sanitized evidence

The 2026-09-14 follow-ups use separate pinned, 32-process comparisons against
takeover revision `ce314e2a`, holding the custom Warp build constant:

- [20260914-scheduling.json](20260914-scheduling.json) records the eight-thread
  scheduling change, measured at `7a156e66`.
- [20260914-fused.json](20260914-fused.json) records scheduling plus fused bound
  initialization, measured at `58ed2eb7`. It includes separately attributed
  truncation-bound fills and the exact reduction in graph kernel launches.

See the [latest report](../CLOTH_SELF_CONTACT_FUSION_REPORT.md) for the result,
validation, and Nsight Compute limitation. The files below describe the
historical comparison.

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
