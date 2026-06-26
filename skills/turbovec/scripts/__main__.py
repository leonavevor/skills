from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from . import benchmark_compare, build_index, integration_scaffold, summarize_results


def _add_common_index_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dim", type=int, default=384, help="vector dimension (default: 384)")
    parser.add_argument("--bit-width", type=int, default=4, help="quantization bits (default: 4)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts",
        description="TurboVec skill helper scripts",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cmp_parser = sub.add_parser(
        "benchmark-compare",
        help="create a markdown benchmark comparison report",
    )
    cmp_parser.add_argument("--input", required=True, help="input JSON path")
    cmp_parser.add_argument("--output", required=True, help="output markdown path")

    scaffold_parser = sub.add_parser(
        "integration-scaffold",
        help="generate a framework integration scaffold",
    )
    scaffold_parser.add_argument(
        "--framework",
        required=True,
        choices=["langchain", "llamaindex", "haystack", "agno"],
        help="target framework",
    )
    scaffold_parser.add_argument("--output", required=True, help="output Python file")

    summarize_parser = sub.add_parser(
        "summarize-results",
        help="generate concise JSON summary from benchmark runs",
    )
    summarize_parser.add_argument("--input", required=True, help="input JSON path")
    summarize_parser.add_argument("--output", required=True, help="output JSON path")

    dir_parser = sub.add_parser(
        "build-from-directory",
        help="build an index from a directory of files",
    )
    dir_parser.add_argument("dirpath", help="directory path")
    _add_common_index_args(dir_parser)
    dir_parser.add_argument("--no-recursive", action="store_true", help="don't recurse subdirectories")
    dir_parser.add_argument("--extensions", help="comma-separated file extensions (e.g., '.txt,.md')")

    file_parser = sub.add_parser(
        "build-from-file",
        help="build an index from a single file or JSONL",
    )
    file_parser.add_argument("filepath", help="file path")
    _add_common_index_args(file_parser)

    urls_parser = sub.add_parser(
        "build-from-urls",
        help="build an index from URLs",
    )
    urls_parser.add_argument("urls", help="comma-separated URLs")
    _add_common_index_args(urls_parser)

    search_parser = sub.add_parser(
        "search",
        help="search an index",
    )
    search_parser.add_argument("index_dir", help="index directory")
    search_parser.add_argument("--query", required=True, help="search query")
    search_parser.add_argument("--k", type=int, default=10, help="number of results (default: 10)")

    sub.add_parser("selftest", help="run minimal built-in self-check")

    return parser


def _run_selftest() -> None:
    sample = {
        "runs": [
            {"name": "faiss", "qps": 1000, "latency_ms_p95": 8.5, "memory_gb": 31.0, "recall_at_10": 0.991},
            {"name": "turbovec", "qps": 1120, "latency_ms_p95": 7.1, "memory_gb": 4.0, "recall_at_10": 0.989},
        ]
    }
    runs = benchmark_compare.parse_runs(sample)
    report = benchmark_compare.generate_markdown(runs)
    assert "TurboVec Benchmark Comparison" in report
    assert "turbovec" in report


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    handlers: dict[str, Callable[[], None]] = {
        "benchmark-compare": lambda: benchmark_compare.run(Path(args.input), Path(args.output)),
        "integration-scaffold": lambda: integration_scaffold.run(args.framework, Path(args.output)),
        "summarize-results": lambda: summarize_results.run(Path(args.input), Path(args.output)),
        "build-from-directory": lambda: build_index.run(
            "build-from-directory",
            dirpath=args.dirpath,
            output=args.output,
            dim=args.dim,
            bit_width=args.bit_width,
            no_recursive=args.no_recursive,
            extensions=args.extensions,
        ),
        "build-from-file": lambda: build_index.run(
            "build-from-file",
            filepath=args.filepath,
            output=args.output,
            dim=args.dim,
            bit_width=args.bit_width,
        ),
        "build-from-urls": lambda: build_index.run(
            "build-from-urls",
            urls=args.urls,
            output=args.output,
            dim=args.dim,
            bit_width=args.bit_width,
        ),
        "search": lambda: build_index.run(
            "search",
            index_dir=args.index_dir,
            query=args.query,
            k=args.k,
        ),
        "selftest": _run_selftest,
    }

    handler = handlers.get(args.command)
    if handler is None:
        raise SystemExit(f"unknown command: {args.command}")
    handler()


if __name__ == "__main__":
    main()

