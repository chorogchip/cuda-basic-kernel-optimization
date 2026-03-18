#!/usr/bin/env python3
import argparse
import math
import os
import re
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


EXEC_RE = re.compile(r'^Executing:\s+(.+?)(?:\s*)$')
# ex:
# 115942030.080560 0.022080 | N = 2^8  = 256
DATA_RE = re.compile(
    r'^\s*'
    r'([0-9]+(?:\.[0-9]+)?)\s+'      # throughput
    r'([0-9]+(?:\.[0-9]+)?)\s*'      # time
    r'\|\s*N\s*=\s*2\^(\d+)\s*=\s*(\d+)'
    r'\s*$'
)

# ex:
#cu8_reduction_add_64_1
BIN_RE = re.compile(r'cu8_reduction_add_(\d+)_(\d+)$')


def parse_log(path: str):
    rows = []
    current_exec = None
    current_block = None
    current_variant = None

    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")

            m_exec = EXEC_RE.match(line)
            if m_exec:
                current_exec = os.path.basename(m_exec.group(1).strip())
                m_bin = BIN_RE.search(current_exec)
                if not m_bin:
                    raise ValueError(
                        f"[line {lineno}] paring failed: {current_exec}\n"
                        f"expected formant: cu8_reduction_add_<block>_<variant>"
                    )
                current_block = int(m_bin.group(1))
                current_variant = int(m_bin.group(2))
                continue

            m_data = DATA_RE.match(line)
            if m_data:
                if current_exec is None:
                    raise ValueError(f"in [line {lineno}] there is no executing line before data line.")

                throughput = float(m_data.group(1))
                time_ms = float(m_data.group(2))
                exp = int(m_data.group(3))
                n = int(m_data.group(4))

                rows.append({
                    "exec_name": current_exec,
                    "block_size": current_block,
                    "variant": current_variant,
                    "throughput": throughput,
                    "time_ms": time_ms,
                    "exp": exp,
                    "n": n,
                })

    if not rows:
        raise ValueError("There is no parsed data.")

    return rows


def save_csv(rows, out_csv):
    fieldnames = [
        "exec_name",
        "block_size",
        "variant",
        "throughput",
        "time_ms",
        "exp",
        "n",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows):
    by_config = defaultdict(list)
    by_n = defaultdict(list)

    for r in rows:
        by_config[(r["block_size"], r["variant"])].append(r)
        by_n[r["n"]].append(r)

    for key in by_config:
        by_config[key].sort(key=lambda x: x["n"])
    for key in by_n:
        by_n[key].sort(key=lambda x: (x["variant"], x["block_size"]))

    return by_config, by_n


def plot_throughput_vs_n(by_config, out_path):
    plt.figure(figsize=(12, 7))

    for (block_size, variant), items in sorted(by_config.items()):
        xs = [r["n"] for r in items]
        ys = [r["throughput"] / 1e9 for r in items]
        label = f"block={block_size}, v={variant}"
        plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=4, label=label)

    plt.xscale("log", base=2)
    plt.xlabel("N")
    plt.ylabel("Throughput (G units/s)")
    plt.title("Reduction throughput vs N")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_time_vs_n(by_config, out_path):
    plt.figure(figsize=(12, 7))

    for (block_size, variant), items in sorted(by_config.items()):
        xs = [r["n"] for r in items]
        ys = [r["time_ms"] for r in items]
        label = f"block={block_size}, v={variant}"
        plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=4, label=label)

    plt.xscale("log", base=2)
    plt.yscale("log", base=10)
    plt.xlabel("N")
    plt.ylabel("Time (ms)")
    plt.title("Reduction time vs N")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_throughput_vs_block_for_each_n(by_n, out_dir, top_k=None):
    ns = sorted(by_n.keys())

    if top_k is not None:
        ns = ns[-top_k:]

    for n in ns:
        items = by_n[n]

        variants = sorted(set(r["variant"] for r in items))
        plt.figure(figsize=(10, 6))

        for variant in variants:
            subset = [r for r in items if r["variant"] == variant]
            subset.sort(key=lambda x: x["block_size"])
            xs = [r["block_size"] for r in subset]
            ys = [r["throughput"] / 1e9 for r in subset]
            plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=5, label=f"v={variant}")

        plt.xscale("log", base=2)
        plt.xlabel("Block size")
        plt.ylabel("Throughput (G units/s)")
        plt.title(f"Throughput vs block size (N={n}, 2^{int(math.log2(n))})")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"throughput_vs_block_N{n}.png"), dpi=160)
        plt.close()


def print_summary(rows):
    print(f"Total rows: {len(rows)}")

    best_by_n = {}
    for r in rows:
        n = r["n"]
        if n not in best_by_n or r["throughput"] > best_by_n[n]["throughput"]:
            best_by_n[n] = r

    print("\nBest config per N:")
    for n in sorted(best_by_n):
        r = best_by_n[n]
        print(
            f"N={n:<9d} "
            f"(2^{r['exp']:<2d})  "
            f"best: block={r['block_size']:<4d} v={r['variant']:<2d}  "
            f"throughput={r['throughput'] / 1e9:.3f} G  "
            f"time={r['time_ms']:.6f} ms"
        )


def main():
    parser = argparse.ArgumentParser(description="Parse CUDA reduction benchmark log and plot graphs.")
    parser.add_argument("logfile", help="Path to log file, e.g. ./log/cu8_result_search.txt")
    parser.add_argument("-o", "--outdir", default="plot_cu8", help="Output directory")
    parser.add_argument("--topk-n", type=int, default=None,
                        help="Only generate throughput-vs-block plots for the largest K N values")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = parse_log(args.logfile)
    by_config, by_n = group_rows(rows)

    csv_path = os.path.join(args.outdir, "parsed.csv")
    save_csv(rows, csv_path)

    plot_throughput_vs_n(by_config, os.path.join(args.outdir, "throughput_vs_n.png"))
    plot_time_vs_n(by_config, os.path.join(args.outdir, "time_vs_n.png"))
    plot_throughput_vs_block_for_each_n(by_n, args.outdir, top_k=args.topk_n)

    print_summary(rows)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved plots under: {args.outdir}")


if __name__ == "__main__":
    main()
