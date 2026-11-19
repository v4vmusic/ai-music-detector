from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

# Standalone scripts (CC §10) get the script directory on sys.path, not the workspace
# root; add the root so `benches` imports resolve when invoked as `python benches/decode.py`.
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from benches._shared.audio import decode_window  # noqa: E402
from benches._shared.constants import (  # noqa: E402
    FILL_MODES,
    PLACEMENT_FIXED,
    PLACEMENTS,
    FillMode,
    Placement,
)
from benches._shared.errors import AidetectError  # noqa: E402

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decode one local audio file into a model window.")
    parser.add_argument("--path", required=True, type=Path, help="local audio file")
    parser.add_argument("--window", required=True, type=float, help="window length in seconds")
    parser.add_argument(
        "--offset",
        default=0.0,
        type=float,
        help="fixed offset in seconds (default: 0; ignored with --placement)",
    )
    parser.add_argument(
        "--placement",
        choices=PLACEMENTS,
        default=None,
        help="per-track placement: start/middle/end are duration-relative; default: fixed offset",
    )
    parser.add_argument(
        "--fill",
        choices=FILL_MODES,
        default="zero-pad",
        help="short-segment fill: zero-pad (baseline) or loop-fill (default: zero-pad)",
    )
    args = parser.parse_args(argv)
    placement = (
        None if args.placement in (None, PLACEMENT_FIXED) else cast(Placement, args.placement)
    )

    try:
        window = decode_window(
            args.path,
            window_seconds=args.window,
            offset_seconds=args.offset,
            fill_mode=cast(FillMode, args.fill),
            placement=placement,
        )
    except (AidetectError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload = {
        "path": str(args.path),
        "sample_rate": window.sample_rate,
        "shape": list(window.samples.shape),
        "duration_seconds": window.samples.size / window.sample_rate,
        "warnings": list(window.warnings),
    }
    sys.stdout.write(json.dumps(payload) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
