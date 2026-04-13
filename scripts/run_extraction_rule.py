from pathlib import Path
import argparse
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kg_extraction import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KG extraction pipeline with rule-based entity extraction")
    parser.add_argument(
        "--input",
        default="",
        help="Optional corpus JSONL path; if omitted, script auto-detects default/fallback paths",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = Path(args.input).resolve() if args.input else None
    run_pipeline(REPO_ROOT, raw_path=raw_path, extraction_method="rule")
    if raw_path:
        print(f"Rule extraction pipeline finished. Input: {raw_path}")
    else:
        print("Rule extraction pipeline finished. Input auto-detected.")
    print("Entity extraction method: rule")
    print("Outputs are in knowledge-project/data/intermediate and knowledge-project/data/output.")


if __name__ == "__main__":
    main()
