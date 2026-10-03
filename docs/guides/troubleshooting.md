# Troubleshooting

[Documentation home](../../README.md)

Use the failing command's error, the campaign logs and `preflight.json` first. Keep failed
campaigns for diagnosis; do not delete inconvenient observations to improve a result.

| Symptom | Action |
|---|---|
| `uv` or Python missing | Follow [setup](getting-started.md); install uv and select Python 3.11. |
| Dependency download fails | Check connectivity/proxy/index access; retry the frozen sync. Do not regenerate the lock merely to bypass a network failure. |
| `nvidia-smi` works, `nvcc` missing | Install the compatible CUDA Toolkit or expose its actual `bin` directory on PATH; see [toolchain](toolchain.md). |
| CUDA build rejects compiler | Check the Toolkit's supported compiler matrix and CMake diagnostics; do not assume the newest host compiler is compatible. |
| Doctor says NVML unavailable | Verify driver/device access in the same environment as `qbl`; Python CUDA packages do not fix a missing host driver. |
| Port already occupied | Identify the owner of ports 11435/8081. Use free configured ports or stop your own idle server; the harness must own its servers. |
| Unresolved toolchain or artifact | Complete pinning/preparation; validate alone only checks configuration structure. |
| Unknown server flag or missing native cache field | The selected engine is incompatible with the adapter. Review pinned source and add a supported adapter change plus contract coverage; never guess zero cache. |
| Runtime settings unverified | Complete the binary-bound source audit and verify context evidence. Setting all booleans true without review is not a fix. |
| OOM, partial offload or no near-limit model | Preserve failure evidence. Revise the draft matrix/common settings and rerun affected pilots; do not silently accept CPU fallback or alter official cells. |
| Cache/token mismatch | Inspect raw native counters, tokenizer IDs, BOS policy and eviction coverage for that prompt/run mode. Resolve before freezing. |
| Monitor overhead exceeds 3% | Investigate paired observations and thermal/work differences; recalibrate the common interval in draft protocol and repeat the pilot. |
| Preflight output already exists | Pick a new pilot directory; preflight deliberately refuses reuse. |
| Frozen file already exists | Use a new filename for a newly passed pilot; do not overwrite prior evidence. |
| Resume identity mismatch | Restore the original environment/config/data, or start a new campaign. Do not edit hashes to force reuse. |
| Journal needs recovery | Resume with the original schedule/environment; the runner acquires the campaign lock and recovers committed writes. Audit again afterward. |
| Missing journal/checksum or changed trace | Restore an intact backup. A corruption failure is not repaired by deleting integrity checks. See [storage compatibility](results.md). |
| Audit exits 2 with verified integrity but incomplete trials | Inspect `n_unresolved` and failures; resume eligible unfinished trials using the unchanged schedule. Exit 0 requires both intact files and complete planned trials. |
| Hardware tests skipped | Set the appropriate real config only after preparation; see [integration instructions](reproduction.md). A skip is not a hardware pass. |
| Matplotlib config directory is unwritable | Set `MPLCONFIGDIR` to a writable local directory before exporting figures. SVG export is also available. |

`qbl audit` verifies stored files and separately reports trial completion. It does not
repeat inference, prove settings from source, or certify GPU equivalence. A failed pilot
may leave valid diagnostic artifacts; those are not official results.
