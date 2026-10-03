# Provenance locks

| File | Purpose | Git policy |
|---|---|---|
| `workloads.json` | Project-authored dataset checksums and provenance | Tracked; update intentionally with dataset changes |
| `toolchain.example.json` | Unresolved template showing the initial toolchain state | Tracked; never a hardware certificate |
| `toolchain.json` | Local binary/native-library hashes, source audit, build cache and compiler identities | Generated and ignored |
| `models.json` | Local source-to-GGUF creation manifests, F16 ancestry and preparation environment | Generated and ignored |

Run `scripts/pin_toolchain.py` to create the local toolchain lock before preparing models.
Do not copy another host's pinned lock: it contains absolute paths and identities for
that build. `scripts/prepare_models.py` then creates/updates the local model index.
Missing or unresolved provenance blocks real inference; synthetic demonstrations can
run without these generated files.

Campaigns snapshot their artifact manifests and toolchain lock inside the raw directory.
Preserve those snapshots for reproduction and review machine paths before sharing.
See [preparation](../docs/guides/toolchain.md) and [results preservation](../docs/guides/results.md).
