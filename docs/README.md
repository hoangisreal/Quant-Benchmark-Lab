# Documentation

[Repository home](../README.md)

## From installation to a reproducible report

1. [Getting started](guides/getting-started.md): install Python dependencies and complete the synthetic demonstration.
2. [Methodology](reference/methodology.md): understand controls, timing, cache, VRAM and quality boundaries.
3. [Toolchain and models](guides/toolchain.md): build the pinned candidate, review controls and prepare GGUF artifacts.
4. [Campaign reproduction](guides/reproduction.md): CPU integration, GPU pilots, freeze, official runs and resume.
5. [Results](guides/results.md): read statistics, preserve raw evidence and regenerate reports offline.

Use [troubleshooting](guides/troubleshooting.md) when a step fails. Check
[engine compatibility](reference/engine-compatibility.md) and
[limitations](reference/limitations.md) before treating a result as a validated comparison.

## Project maintenance

- [Development](development.md): architecture, configuration, conventions and checks.
- [Audit and research, 2026-10-03](reviews/2026-10-03.md): inspected scope, fixes, methodology limits and remaining evidence.
- [Configuration layout](../configs/README.md) and [provenance locks](../locks/README.md).

`guides/` contains procedures; `reference/` contains measurement definitions and evidence;
`reviews/` contains public audit summaries. Internal AI-agent notes remain local under
an ignored directory and are not required to follow these guides.

## Next lab milestone

Finish Qwen3-4B artifact preparation and verify E3 lineage. Then run GPU lifecycle checks
and a passing pilot for each matrix on the RTX 3050 4GB. Only passing, matching certificates
permit freeze and official campaigns. No official GPU result is currently published.
