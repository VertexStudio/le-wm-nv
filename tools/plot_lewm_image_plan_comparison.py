#!/usr/bin/env python3
"""Render a LeWM image-input Python-vs-Rust planning benchmark as SVG."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


DEFAULT_ROWS = [
    ("current_decode_preprocess", "current media"),
    ("goal_decode_preprocess", "goal media"),
    ("current_encode", "current encode"),
    ("goal_encode", "goal encode"),
    ("planning", "iCEM planning"),
    ("selected_score", "selected score"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True, type=Path, dest="python_json")
    parser.add_argument("--rust", required=True, type=Path, dest="rust_json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="LeWM PushT Image Planning Latency")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = json.loads(args.python_json.read_text(encoding="utf-8"))
    rust = json.loads(args.rust_json.read_text(encoding="utf-8"))
    rows = []
    for key, label in DEFAULT_ROWS:
        py = benchmark_value_ms(python, key)
        rs = benchmark_value_ms(rust, key)
        rows.append((label, py, rs, py / rs if rs > 0 else 0.0))

    subtitle = (
        f"Metric: synchronized CUDA wall time in ms. "
        f"Planner={rust.get('planner', 'icem')}; samples={rust.get('samples')}; "
        f"iterations={rust.get('iterations')}; horizon={rust.get('horizon')}. Lower is faster."
    )
    svg = render_svg(args.title, subtitle, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"wrote {args.output}")


def benchmark_value_ms(payload: dict, key: str) -> float:
    stats = payload.get("benchmark_stats", {}).get(key)
    if stats is not None and "p50_ms" in stats:
        return float(stats["p50_ms"])
    return float(payload["timing_ms"][key])


def render_svg(
    title: str,
    subtitle: str,
    rows: list[tuple[str, float, float, float]],
) -> str:
    width = 1120
    left = 230
    right = 240
    top = 96
    row_h = 62
    chart_w = width - left - right
    height = top + 42 + row_h * len(rows)
    max_value = max(max(py, rs) for _, py, rs, _ in rows)
    scale = chart_w / max_value if max_value > 0 else 1.0

    parts = [
        svg_open(width, height),
        "<style>",
        "text{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#172026}",
        ".title{font-size:28px;font-weight:760}.sub{font-size:14px;fill:#52606d}.label{font-size:15px;font-weight:650}",
        ".value{font-size:13px;fill:#303b45}.axis{stroke:#d7dde3;stroke-width:1}.py{fill:#c2410c}.rs{fill:#0f766e}",
        ".grid{stroke:#edf1f5;stroke-width:1}.speed{font-size:13px;font-weight:700;fill:#172026}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="28" y="38">{html.escape(title)}</text>',
        f'<text class="sub" x="28" y="62">{html.escape(subtitle)}</text>',
        f'<text class="sub" x="{left}" y="86">Python / PyTorch</text>',
        f'<rect class="py" x="{left + 125}" y="74" width="14" height="14" rx="2"/>',
        f'<text class="sub" x="{left + 155}" y="86">Rust / Candle</text>',
        f'<rect class="rs" x="{left + 250}" y="74" width="14" height="14" rx="2"/>',
    ]

    ticks = 4
    for tick in range(ticks + 1):
        value = max_value * tick / ticks
        x = left + value * scale
        parts.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - 28}"/>'
        )
        parts.append(
            f'<text class="sub" x="{x:.1f}" y="{height - 10}" text-anchor="middle">{value:.0f}ms</text>'
        )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{height - 28}" x2="{left + chart_w}" y2="{height - 28}"/>'
    )

    for idx, (label, py, rs, speedup) in enumerate(rows):
        y = top + idx * row_h
        py_w = py * scale
        rs_w = rs * scale
        py_value_x, py_anchor = value_label(left, chart_w, py_w)
        rs_value_x, rs_anchor = value_label(left, chart_w, rs_w)
        parts.extend(
            [
                f'<text class="label" x="28" y="{y + 27}">{html.escape(label)}</text>',
                f'<rect class="py" x="{left}" y="{y + 8}" width="{py_w:.1f}" height="18" rx="4"/>',
                f'<rect class="rs" x="{left}" y="{y + 32}" width="{rs_w:.1f}" height="18" rx="4"/>',
                f'<text class="value" x="{py_value_x:.1f}" y="{y + 22}" text-anchor="{py_anchor}">{py:.3f} ms</text>',
                f'<text class="value" x="{rs_value_x:.1f}" y="{y + 46}" text-anchor="{rs_anchor}">{rs:.3f} ms</text>',
                f'<text class="speed" x="{left + chart_w + 24}" y="{y + 35}">{speedup:.2f}x</text>',
            ]
        )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def svg_open(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img">'
    )


def value_label(left: int, chart_w: int, bar_w: float) -> tuple[float, str]:
    outside_x = left + bar_w + 8
    if outside_x <= left + chart_w - 74:
        return outside_x, "start"
    return left + bar_w - 8, "end"


if __name__ == "__main__":
    main()
