from pathlib import Path
import sys
import argparse


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kg_extraction import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KG extraction pipeline")
    parser.add_argument(
        "--input",
        default="",
        help="Optional corpus JSONL path; if omitted, script auto-detects default/fallback paths",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = Path(args.input).resolve() if args.input else None
    run_pipeline(REPO_ROOT, raw_path=raw_path)
    if raw_path:
        print(f"Extraction pipeline finished. Input: {raw_path}")
    else:
        print("Extraction pipeline finished. Input auto-detected.")
    print("Outputs are in knowledge-project/data/intermediate and knowledge-project/data/output.")


if __name__ == "__main__":
    main()
