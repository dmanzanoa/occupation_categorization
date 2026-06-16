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
        ROOT / "data" / "codes_enriched.json", use_semantic=arguments.semantic
    )
    for position, result in enumerate(
        matcher.search(arguments.description, arguments.top_k), start=1
    ):
        print(f"{position}. {result.label} [{result.code}] - {result.score:.3f}")
        if result.ciuo08_code:
            print(f"   CIUO-08: {result.ciuo08_code} - {result.ciuo08_label}")
        if result.ciuo08_cl_path:
            print(f"   CIUO-08 CL: {result.ciuo08_cl_path}")
        print(f"   Códigos unidos: {result.joined_codes}")
        print(f"   {result.description}\n")


if __name__ == "__main__":
    main()
