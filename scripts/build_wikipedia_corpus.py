import argparse
import json
from pathlib import Path
from typing import Dict, List

try:
    import wikipedia  # type: ignore[import-not-found]
    from wikipedia.exceptions import DisambiguationError, PageError  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'wikipedia'. Install with: pip install wikipedia") from exc


DEFAULT_TURING_PAGES = [
    "Alan Turing",
    "Turing machine",
    "Turing test",
    "Church-Turing thesis",
    "Bletchley Park",
    "Enigma machine",
    "On Computable Numbers",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"


def split_text_by_paragraph(content: str, min_chars: int = 80) -> List[str]:
    chunks: List[str] = []
    for part in content.split("\n"):
        text = part.strip()
        if len(text) >= min_chars:
            chunks.append(text)
    return chunks


def fetch_page(title: str, source: str) -> List[Dict[str, str]]:
    try:
        page = wikipedia.page(title, auto_suggest=False)
    except DisambiguationError as exc:
        if not exc.options:
            print(f"[WARN] Disambiguation without options: {title}")
            return []
        fallback = exc.options[0]
        print(f"[WARN] Disambiguation: {title} -> use {fallback}")
        page = wikipedia.page(fallback, auto_suggest=False)
    except PageError:
        print(f"[WARN] Page not found: {title}")
        return []

    paragraphs = split_text_by_paragraph(page.content)
    docs: List[Dict[str, str]] = []
    for idx, paragraph in enumerate(paragraphs, start=1):
        docs.append(
            {
                "doc_id": f"wiki_{page.title.replace(' ', '_').lower()}_{idx:03d}",
                "source": source,
                "title": page.title,
                "url": page.url,
                "text": paragraph,
            }
        )
    return docs


def save_jsonl(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Turing corpus from Wikipedia")
    parser.add_argument(
        "--titles",
        nargs="*",
        default=DEFAULT_TURING_PAGES,
        help="Wikipedia page titles to fetch",
    )
    parser.add_argument(
        "--filename",
        default="turing_corpus.jsonl",
        help="Output filename under knowledge-project/data/raw",
    )
    parser.add_argument("--lang", default="en", help="Wikipedia language")
    parser.add_argument("--source", default="wikipedia", help="Source label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wikipedia.set_lang(args.lang)

    all_rows: List[Dict[str, str]] = []
    for title in args.titles:
        print(f"[INFO] Fetching: {title}")
        rows = fetch_page(title, source=args.source)
        all_rows.extend(rows)

    output_path = RAW_DIR / args.filename
    save_jsonl(all_rows, output_path)
    print(f"[DONE] Saved {len(all_rows)} records to {output_path}")


if __name__ == "__main__":
    main()
