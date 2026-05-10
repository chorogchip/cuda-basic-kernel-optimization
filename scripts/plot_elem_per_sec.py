#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
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
    r"(?:impl=(?P<impl>\S+)\s+)?"
    r".*?target=(?P<target>\S+)\s+"
    r"N\s+(?P<n>\d+)\s+"
    r"LOGN\s+(?P<logn>\d+)\s+"
    r"elem_per_sec\s+(?P<elem_per_sec>\S+)\s+"
    r"TOTALMS\s+(?P<totalms>\S+)\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot elem_per_sec vs LOGN from one or more *_run.txt or *_run_maxperf.txt files."
    )
    parser.add_argument("inputs", nargs="+", help="Input *_run.txt or *_run_maxperf.txt files")
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
                    "impl": match.group("impl") or "my",
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
    is_maxperf_plot = all(path.name.endswith("_run_maxperf.txt") for path in paths)

    if len(paths) == 1:
        stem = paths[0].name.removesuffix(".txt")
        return plots_dir / f"{stem}_plot.png"

    if is_maxperf_plot:
        return plots_dir / f"{kernel}_maxperf_plot.png"

    name_prefix = Path(paths[0]).name.split("_run")[0]
    if all(path.name.startswith(name_prefix) for path in paths[1:]):
        return plots_dir / f"{name_prefix}_plot.png"

    return plots_dir / f"{kernel}_combined_plot.png"


def split_target(kernel: str, target: str) -> list[str]:
    if target == kernel:
        return []

    prefix = f"{kernel}_"
    if target.startswith(prefix):
        return target[len(prefix) :].split("_")

    return target.split("_")


def read_kernel_meta(kernel: str) -> dict[str, object]:
    meta: dict[str, object] = {
        "versioned": False,
        "params": [],
        "dram_bytes_per_elem": None,
        "l2_working_set_bytes_per_unit": None,
        "l2_problem_power": 1.0,
    }
    path = Path("kernels") / kernel / "plot_meta.txt"
    if not path.is_file():
        return meta

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key == "versioned":
                meta["versioned"] = value.lower() in {"1", "true", "yes"}
            elif key == "params":
                meta["params"] = [item for item in value.split() if item]
            elif key == "dram_bytes_per_elem":
                meta["dram_bytes_per_elem"] = float(value)
            elif key == "l2_working_set_bytes_per_unit":
                meta["l2_working_set_bytes_per_unit"] = float(value)
            elif key == "l2_problem_power":
                meta["l2_problem_power"] = float(value)

    return meta


def read_plot_constants() -> dict[str, str]:
    constants: dict[str, str] = {}
    path = Path("notes") / "plot_constants.txt"
    if not path.is_file():
        return constants

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            constants[key.strip()] = value.strip()

    return constants


def dram_bytes_per_sec(constants: dict[str, str]) -> float | None:
    if "dram_bandwidth_bytes_per_sec" in constants:
        return float(constants["dram_bandwidth_bytes_per_sec"])
    if "dram_bandwidth_gib_per_sec" in constants:
        return float(constants["dram_bandwidth_gib_per_sec"]) * 1024.0**3
    if "dram_bandwidth_gbps" in constants:
        return float(constants["dram_bandwidth_gbps"]) * 1.0e9
    return None


def read_baseline_label(kernel: str) -> str:
    path = Path("kernels") / kernel / "baseline.txt"
    if not path.is_file():
        return "baseline"

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return f"baseline ({stripped})"

    return "baseline"


def build_series_name(
    row: dict[str, object],
    is_maxperf_plot: bool,
    baseline_label: str,
    kernel_meta: dict[str, object],
) -> str:
    impl = str(row["impl"])
    target = str(row["target"])
    kernel = str(row["kernel"])

    if impl == "baseline":
        return baseline_label

    if not is_maxperf_plot:
        return f"{impl}:{target}"

    parts = split_target(kernel, target)
    if kernel_meta["versioned"] and parts:
        return f"version {parts[0]}"

    return impl


def build_point_label(row: dict[str, object], is_maxperf_plot: bool, kernel_meta: dict[str, object]) -> str:
    if not is_maxperf_plot or row["impl"] == "baseline":
        return ""

    kernel = str(row["kernel"])
    target = str(row["target"])
    parts = split_target(kernel, target)
    params = list(kernel_meta["params"])

    if kernel_meta["versioned"] and parts:
        parts = parts[1:]

    if not parts:
        return ""

    return ",".join(parts)


def main() -> int:
    args = parse_args()
    input_paths = [Path(item) for item in args.inputs]
    is_maxperf_plot = all(path.name.endswith("_run_maxperf.txt") for path in input_paths)

    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        if not (path.name.endswith("_run.txt") or path.name.endswith("_run_maxperf.txt")):
            raise ValueError(f"Expected a *_run.txt or *_run_maxperf.txt file: {path}")

    parsed_rows: list[dict[str, object]] = []
    for path in input_paths:
        parsed_rows.extend(parse_run_file(path))

    kernels = {row["kernel"] for row in parsed_rows}
    if len(kernels) != 1:
        raise ValueError(f"All inputs must belong to one kernel family, got: {sorted(kernels)}")

    kernel = next(iter(kernels))
    kernel_meta = read_kernel_meta(str(kernel))
    plot_constants = read_plot_constants()
    baseline_label = read_baseline_label(str(kernel))
    series: dict[str, dict[int, float]] = defaultdict(dict)
    point_labels: dict[tuple[str, int], str] = {}

    for row in parsed_rows:
        target = build_series_name(row, is_maxperf_plot, baseline_label, kernel_meta)
        logn = int(row["logn"])
        elem_per_sec = float(row["elem_per_sec"])

        if logn in series[target]:
            raise ValueError(f"Duplicate LOGN={logn} for target={target}")

        series[target][logn] = elem_per_sec
        point_labels[(target, logn)] = build_point_label(row, is_maxperf_plot, kernel_meta)

    output_path = build_output_path(input_paths, kernel)

    all_logn_values = sorted({int(row["logn"]) for row in parsed_rows})

    fig, ax = plt.subplots(figsize=(14, 9))

    sorted_targets = sorted(series)
    label_offsets = {}
    if is_maxperf_plot:
        custom_targets = [target for target in sorted_targets if not target.startswith("baseline")]
        for idx, target in enumerate(custom_targets):
            direction = 1 if idx % 2 == 0 else -1
            step = (idx // 2) + 1
            label_offsets[target] = (0, direction * (8 + 10 * (step - 1)))

    for target in sorted_targets:
        points = sorted(series[target].items())
        x_values = [item[0] for item in points]
        y_values = [item[1] for item in points]
        ax.plot(x_values, y_values, marker="o", linewidth=1.8, markersize=4, label=target)

        if is_maxperf_plot:
            for x_value, y_value in points:
                label = point_labels.get((target, x_value), "")
                if label:
                    xytext = label_offsets.get(target, (0, 6))
                    va = "bottom" if xytext[1] >= 0 else "top"
                    ax.annotate(
                        label,
                        (x_value, y_value),
                        textcoords="offset points",
                        xytext=xytext,
                        ha="center",
                        va=va,
                        fontsize=7,
                        rotation=30,
                    )

    dram_bytes_per_elem = kernel_meta["dram_bytes_per_elem"]
    dram_bps = dram_bytes_per_sec(plot_constants)
    if dram_bytes_per_elem is not None and dram_bps is not None:
        ref_elem_per_sec = dram_bps / float(dram_bytes_per_elem)
        dram_label = "DRAM BW"
        if "dram_bandwidth_label" in plot_constants:
            dram_label += f" {plot_constants['dram_bandwidth_label']}"
        ax.axhline(
            float(ref_elem_per_sec),
            linestyle="--",
            linewidth=1.2,
            color="black",
            alpha=0.55,
            label=dram_label,
        )

    l2_bytes_per_unit = kernel_meta["l2_working_set_bytes_per_unit"]
    l2_cache_bytes = plot_constants.get("l2_cache_bytes")
    if l2_bytes_per_unit is not None and l2_cache_bytes is not None:
        l2_problem_power = float(kernel_meta["l2_problem_power"])
        l2_units = float(l2_cache_bytes) / float(l2_bytes_per_unit)
        l2_logn = math.log2(l2_units) / l2_problem_power
        l2_label = "L2"
        if "l2_cache_label" in plot_constants:
            l2_label += f" {plot_constants['l2_cache_label']}"
        ax.axvline(
            float(l2_logn),
            linestyle=":",
            linewidth=1.2,
            color="black",
            alpha=0.55,
            label=l2_label,
        )

    ax.set_title(f"{kernel} elem_per_sec")
    ax.set_xlabel("LOGN")
    ax.set_ylabel("elem_per_sec")
    ax.set_xticks(all_logn_values)
    ax.set_xticklabels([str(value) for value in all_logn_values])
    ax.set_xlim(min(all_logn_values), max(all_logn_values))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.margins(x=0.02, y=0.08)
    ax.grid(True, linestyle="--", alpha=0.35)

    params = list(kernel_meta["params"])
    if is_maxperf_plot and params:
        ax.text(
            1.02,
            0.02,
            f"point label: {', '.join(params)}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            family="monospace",
        )

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
