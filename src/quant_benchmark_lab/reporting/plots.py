"""Static per-prompt SVG plots; PNG exports when Matplotlib is available."""

from html import escape
from pathlib import Path


def charts(rows: list[dict], output: Path, synthetic: bool) -> list[str]:
    files = []
    pairs = sorted({(r["prompt_id"], r["run_mode"]) for r in rows})
    for prompt, mode in pairs:
        for metric in ("ttft_stream_ms", "decode_tok_s", "peak_request_vram_bytes"):
            chosen = [r for r in rows if r["prompt_id"] == prompt and r["run_mode"] == mode
                      and r["metric"] == metric and r["median"] is not None]
            if not chosen:
                continue
            label = f"{'SYNTHETIC — ' if synthetic else ''}{prompt} / {mode} / {metric}"
            maximum = max(r["median"] for r in chosen) or 1
            height = 100 + 45 * len(chosen)
            elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="20" y="25" font-size="16">{escape(label)}</text>',
                '<text x="20" y="48" font-size="12">Median per prompt; n = valid observations. CSV includes mean and sample std.</text>']
            for i, r in enumerate(chosen):
                y = 70 + i * 45
                width = r["median"] / maximum * 460
                value = r["median"] / 1024**2 if r["unit"] == "bytes" else r["median"]
                unit = "MiB" if r["unit"] == "bytes" else r["unit"]
                elements += [f'<text x="20" y="{y + 18}">{escape(r["cell_id"])}</text>',
                             f'<rect x="235" y="{y}" width="{width:.2f}" height="25" fill="#2563eb"/>',
                             f'<text x="710" y="{y + 18}">{value:.2f} {unit}; n={r["n"]}</text>']
            name = f"{prompt}-{mode}-{metric}.svg"
            (output / name).write_text("\n".join(elements + ["</svg>"]), encoding="utf-8")
            files.append(name)
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(9, 3))
                ax.bar([r["cell_id"] for r in chosen], [r["median"] for r in chosen])
                ax.set_title(label)
                ax.set_ylabel(chosen[0]["unit"])
                fig.tight_layout()
                png = name.replace(".svg", ".png")
                fig.savefig(output / png)
                plt.close(fig)
                files.append(png)
            except ImportError:
                pass  # SVG remains a standalone scientific artifact; missing optional export is explicit.
    return files


def tradeoff_charts(rows, quality, output: Path, synthetic: bool) -> list[str]:
    scores = {r["cell_id"]: r["macro_accuracy"] for r in quality.get("summary", [])}
    files = []
    for prompt, mode in sorted({(r["prompt_id"], r["run_mode"]) for r in rows}):
        for metric in ("decode_tok_s", "peak_request_vram_bytes"):
            points = [r for r in rows if r["prompt_id"] == prompt and r["run_mode"] == mode
                      and r["metric"] == metric and r["median"] is not None and r["cell_id"] in scores]
            if not points:
                continue
            scale = 1024**2 if metric.endswith("bytes") else 1
            unit = "MiB" if scale != 1 else "native tok/s"
            maximum = max(r["median"] / scale for r in points) * 1.1 or 1
            title = f"{'SYNTHETIC — ' if synthetic else ''}Quality vs {metric} / {prompt} / {mode}"
            svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="450">',
                   '<rect width="100%" height="100%" fill="white"/>',
                   f'<text x="30" y="25" font-size="16">{escape(title)}</text>',
                   '<path d="M100 70 V350 H820" stroke="black" fill="none"/>',
                   '<text x="10" y="60">Accuracy</text>', '<text x="70" y="75">1.0</text>',
                   '<text x="70" y="350">0.0</text>',
                   f'<text x="350" y="400">Median {unit} on this prompt (no error bars)</text>',
                   f'<text x="750" y="375">{maximum:.2f}</text>',
                   '<text x="100" y="375">0</text>',
                   '<text x="30" y="430" font-size="12">Quality is macro task accuracy; tokenizers differ across models. See per-trial counts and failures.</text>']
            for i, r in enumerate(points):
                x = 100 + r["median"] / scale / maximum * 720
                y = 350 - scores[r["cell_id"]] * 280
                svg += [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#2563eb"/>',
                        f'<text x="{max(110, x-100):.2f}" y="{y-12-i*16:.2f}" font-size="12">'
                        f'{escape(r["cell_id"])}; n={r["n"]}</text>']
            name = f"{prompt}-{mode}-quality-vs-{metric}.svg"
            (output / name).write_text("\n".join(svg + ["</svg>"]), encoding="utf-8")
            files.append(name)
    return files
