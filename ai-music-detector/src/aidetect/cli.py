from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any
from urllib.parse import urlparse

from aidetect.detector import CascadeDetector, Detector
from aidetect.errors import AidetectError

__all__ = ["main"]

_URL_SCHEMES = frozenset({"http", "https"})

_EXAMPLES = (
    "examples:\n"
    "  aidetect score track.mp3 --json\n"
    "  aidetect score https://example.com/track.mp3 --json\n"
    "  aidetect score track.mp3 --cascade --json"
)

_JSON_HELP = "emit exactly one JSON object on stdout (default: short label/probability line)"
_CASCADE_HELP = "use the D1 two-stage conservative policy (nested primary/second_opinion in JSON)"

_USAGE_ORDER = (
    "usage order:\n"
    "  aidetect score <source> [flags]\n"
    "Flags belong to the score command and must come after `score`; within that command\n"
    "they may appear before or after <source>. Root-level flags are rejected:\n"
    "  aidetect score track.mp3 --json --cascade   # ok\n"
    "  aidetect --json score track.mp3            # rejected"
)

_SCORING_FLAGS = f"scoring flags:\n  --json      {_JSON_HELP}\n  --cascade   {_CASCADE_HELP}"

_OUTPUT_CONTRACT = (
    "JSON goes to stdout; diagnostics and errors go to stderr.\n"
    "Errors exit 1 (argument errors exit 2)."
)


def _is_url(source: str) -> bool:
    return urlparse(source).scheme.lower() in _URL_SCHEMES


def _diagnose(message: str) -> None:
    sys.stderr.write(f"aidetect: {message}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aidetect",
        description=(
            "AI-generated music detection with calibrated tri-state labels "
            "(human, uncertain, ai_generated).\n"
            "Score one local audio file or an http(s) URL with the `score` command."
        ),
        epilog=(
            f"{_USAGE_ORDER}\n\n"
            f"{_SCORING_FLAGS}\n\n"
            f"{_EXAMPLES}\n\n"
            f"{_OUTPUT_CONTRACT}\n"
            "Run 'aidetect score --help' for the scoring help screen."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser(
        "score",
        help="score one local audio file or http(s) URL",
        description="Score one local audio file or http(s) URL.",
        epilog=f"{_EXAMPLES}\n\n{_OUTPUT_CONTRACT}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    score.add_argument("source", help="local audio path or http(s) URL")
    score.add_argument("--json", action="store_true", dest="as_json", help=_JSON_HELP)
    score.add_argument("--cascade", action="store_true", help=_CASCADE_HELP)
    return parser


def _summarize(payload: dict[str, Any]) -> str:
    # Any: dataclasses.asdict returns dynamically typed nested dictionaries.
    label = payload["label"]
    primary = payload.get("primary")
    nested = primary if isinstance(primary, dict) else payload
    return f"{label}\tprobability={float(nested['probability']):.4f}"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: JSON on stdout, diagnostics and errors on stderr."""

    args = _build_parser().parse_args(argv)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if args.cascade:
                detector: Detector | CascadeDetector = CascadeDetector.from_pretrained()
            else:
                detector = Detector.from_pretrained()
            if _is_url(args.source):
                result = detector.score_url(args.source)
            else:
                result = detector.score_file(args.source)
        for warning in caught:
            _diagnose(f"{warning.category.__name__}: {warning.message}")
    except (AidetectError, ValueError, OSError) as exc:
        _diagnose(str(exc))
        return 1

    payload = asdict(result)
    if args.as_json:
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    else:
        sys.stdout.write(_summarize(payload) + "\n")
    return 0
