from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from esco_matcher import EscoMatcher  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Buscar una ocupación ESCO en español")
    parser.add_argument("description", help="Descripción libre de la ocupación")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--semantic", action="store_true")
    arguments = parser.parse_args()

    matcher = EscoMatcher(
        ROOT / "data" / "data_oc.csv", use_semantic=arguments.semantic
    )
    for position, result in enumerate(
        matcher.search(arguments.description, arguments.top_k), start=1
    ):
        print(f"{position}. {result.label} [{result.code}] - {result.score:.3f}")
        print(f"   {result.description}\n")


if __name__ == "__main__":
    main()
