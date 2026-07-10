#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy>=2.0",
#     "plotly>=5.24",
#     "polars>=1.0",
# ]
# ///
"""
Chart one or more swiv --mem-usage-csv-path recordings with Plotly.

The csv format is headerless "wallclock-millisecond-ts,allocated-in-bytes"
lines, sampled every 10ms. Each file becomes one series, so before/after
comparisons are just a matter of passing both files. By default the x-axis is
seconds since each run started, which lines runs up for comparison; pass
--wallclock to plot against absolute time instead.

Examples
--------
    # Single run (pops up in your browser):
    ./scripts/plot_mem_usage.py mem.csv

    # Compare two runs, also saved to HTML:
    ./scripts/plot_mem_usage.py before.csv after.csv -o comparison.html
"""

import argparse
import sys
from pathlib import Path

import plotly.express as px
import polars as pl


def load(path: Path, wallclock: bool) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        has_header=False,
        new_columns=["ts_ms", "bytes"],
        schema={"ts_ms": pl.Int64, "bytes": pl.Int64},
    )

    if wallclock:
        x = pl.from_epoch("ts_ms", time_unit="ms").alias("_x")
    else:
        x = ((pl.col("ts_ms") - pl.col("ts_ms").min()) / 1000).alias("_x")

    return df.with_columns(
        x,
        (pl.col("bytes") / 2**20).alias("allocated_mib"),
        pl.lit(path.stem).alias("run"),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Chart swiv --mem-usage-csv-path recordings with Plotly."
    )
    p.add_argument("csv", nargs="+", type=Path, help="One csv per run to chart.")
    p.add_argument("-t", "--title", default="swiv jemalloc allocated bytes")
    p.add_argument("-o", "--out", default=None, help="Also save the chart to this HTML file.")
    p.add_argument(
        "--wallclock",
        action="store_true",
        help="Plot against absolute time instead of seconds since run start.",
    )
    args = p.parse_args(argv)

    df = pl.concat(load(path, args.wallclock) for path in args.csv)
    print(f"Parsed {df.height} samples from {len(args.csv)} run(s).", file=sys.stderr)

    fig = px.line(
        df,
        x="_x",
        y="allocated_mib",
        color="run",
        labels={
            "_x": "Time" if args.wallclock else "Elapsed (s)",
            "allocated_mib": "Allocated (MiB)",
        },
    )
    fig.update_layout(
        title=args.title,
        template="plotly_white",
        hovermode="x unified",
    )

    if args.out:
        out = Path(args.out)
        fig.write_html(out, include_plotlyjs="cdn")
        print(f"Wrote chart -> {out.resolve()}", file=sys.stderr)

    fig.show()  # pops up the interactive chart in your default browser
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
