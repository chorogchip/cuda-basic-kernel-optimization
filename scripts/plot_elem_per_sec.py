#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


LINE_RE = re.compile(
    r"^(?P<kernel>\S+)\s+"
    r".*?target=(?P<target>\S+)\s+"
    r"N\s+(?P<n>\d+)\s+"
    r"LOGN\s+(?P<logn>\d+)\s+"
    r"elem_per_sec\s+(?P<elem_per_sec>\S+)\s+"
    r"TOTALMS\s+(?P<totalms>\S+)\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot elem_per_sec vs LOGN from one or more *_run.txt files."
    )
    parser.add_argument("inputs", nargs="+", help="Input *_run.txt files")
    return parser.parse_args()


def parse_run_file(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            match = LINE_RE.match(stripped)
            if match is None:
                raise ValueError(f"Could not parse {path}:{lineno}")

            rows.append(
                {
                    "kernel": match.group("kernel"),
                    "target": match.group("target"),
                    "n": int(match.group("n")),
                    "logn": int(match.group("logn")),
                    "elem_per_sec": float(match.group("elem_per_sec")),
                }
            )

    if not rows:
        raise ValueError(f"No benchmark rows found in {path}")

    return rows


def build_output_path(paths: list[Path], kernel: str) -> Path:
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    if len(paths) == 1:
        stem = paths[0].name.removesuffix(".txt")
        return plots_dir / f"{stem}_plot.png"

    name_prefix = Path(paths[0]).name.split("_run")[0]
    if all(path.name.startswith(name_prefix) for path in paths[1:]):
        return plots_dir / f"{name_prefix}_plot.png"

    return plots_dir / f"{kernel}_combined_plot.png"


def main() -> int:
    args = parse_args()
    input_paths = [Path(item) for item in args.inputs]

    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        if not path.name.endswith("_run.txt"):
            raise ValueError(f"Expected a *_run.txt file: {path}")

    parsed_rows: list[dict[str, object]] = []
    for path in input_paths:
        parsed_rows.extend(parse_run_file(path))

    kernels = {row["kernel"] for row in parsed_rows}
    if len(kernels) != 1:
        raise ValueError(f"All inputs must belong to one kernel family, got: {sorted(kernels)}")

    kernel = next(iter(kernels))
    series: dict[str, dict[int, float]] = defaultdict(dict)

    for row in parsed_rows:
        target = str(row["target"])
        logn = int(row["logn"])
        elem_per_sec = float(row["elem_per_sec"])

        if logn in series[target]:
            raise ValueError(f"Duplicate LOGN={logn} for target={target}")

        series[target][logn] = elem_per_sec

    output_path = build_output_path(input_paths, kernel)

    all_logn_values = sorted({int(row["logn"]) for row in parsed_rows})

    fig, ax = plt.subplots(figsize=(14, 9))

    for target in sorted(series):
        points = sorted(series[target].items())
        x_values = [item[0] for item in points]
        y_values = [item[1] for item in points]
        ax.plot(x_values, y_values, marker="o", linewidth=1.8, markersize=4, label=target)

    ax.set_title(f"{kernel} elem_per_sec")
    ax.set_xlabel("N")
    ax.set_ylabel("elem_per_sec")
    ax.set_xticks(all_logn_values)
    ax.set_xticklabels([f"2^{value}" for value in all_logn_values])
    ax.set_xlim(min(all_logn_values), max(all_logn_values))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.margins(x=0.02, y=0.08)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    fig.savefig(output_path, dpi=160)

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
