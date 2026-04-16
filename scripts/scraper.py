import argparse
import csv
import json
import re
import time
import warnings
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

# 导入维基百科库
try:
    import wikipedia  # type: ignore[import-not-found]
    from wikipedia.exceptions import DisambiguationError, PageError  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'wikipedia'. Install with: pip install wikipedia") from exc

warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

# 无效链接的过滤前缀
IGNORE_PREFIXES = (
    "Category:", "Template:", "List of", "Wikipedia:", 
    "Portal:", "Help:", "Talk:", "Draft:", "User:"
)

SCHEMA_SEEDS: Dict[str, List[str]] = {
    "Person": ["Alan Turing", "Joan Clarke", "Alonzo Church", "John von Neumann", "Claude Shannon", "Max Newman", "Donald Michie", "I. J. Good", "Gordon Welchman", "Hugh Alexander", "Robin Gandy"],
    "Organization": ["King's College, Cambridge", "University of Cambridge", "Princeton University", "Bletchley Park", "University of Manchester", "Government Code and Cypher School", "National Physical Laboratory (United Kingdom)", "Hut 8", "Bell Labs", "Foreign Office"],
    "Location": ["London", "Manchester", "Princeton, New Jersey", "Maida Vale", "Wilmslow", "Bletchley", "Cambridge", "Sherborne School"],
    "Concept": ["Turing machine", "Turing test", "Halting problem", "Church-Turing thesis", "Computability theory", "Morphogenesis", "Universal Turing machine", "Decision problem", "Oracle machine", "Imitation game", "Turing reduction"],
    "Artifact": ["Bombe", "Automatic Computing Engine", "Enigma machine", "Manchester Baby", "Colossus computer", "Pilot ACE", "Universal machine"],
    "Event": ["World War II", "Cryptanalysis of the Enigma", "Alan Turing law", "Royal pardon", "Turing centenary", "Prosecution of Alan Turing", "Second Boer War"],
    "Publication": ["On Computable Numbers", "Computing Machinery and Intelligence", "Systems of Logic Based on Ordinals", "The Chemical Basis of Morphogenesis", "Intelligent Machinery", "Can Digital Computers Think?", "The Applications of Probability to Cryptography"],
    "Honor": ["Turing Award", "Order of the British Empire", "Bank of England 50 note", "Alan Turing Year", "Turing's law"],
}

@dataclass
class PageFetchResult:
    seed_label: str
    seed_title: str
    requested_title: str
    final_title: str
    url: str
    status: str
    paragraph_count: int


def slugify(text: str) -> str:
    """精简：利用正则快速过滤非字母数字字符"""
    return re.sub(r'[^a-z0-9]+', '_', text.strip().lower()).strip('_')


def split_text_by_paragraph(content: str, min_chars: int) -> List[str]:
    """精简与优化：过滤掉长度不足的段落，并清洗类似 '== Early life ==' 的维基标题"""
    parts = []
    for block in content.split("\n"):
        text = block.strip()
        # 跳过长度不够的段落以及维基百科的章节标题
        if len(text) >= min_chars and not re.match(r"^=+\s*[^=]+\s*=+$", text):
            parts.append(text)
    return parts


def iter_seed_titles(custom_titles: List[str] | None) -> Iterable[Tuple[str, str]]:
    if custom_titles:
        yield from (("Unknown", title) for title in custom_titles)
    else:
        for label, titles in SCHEMA_SEEDS.items():
            for title in titles:
                yield label, title


def safe_fetch_page_and_content(title: str) -> Tuple[object | None, str, str, str, str]:
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            page = wikipedia.page(title, auto_suggest=False)
            final_title = str(getattr(page, "title", title))
            url = str(getattr(page, "url", ""))
            content = str(getattr(page, "content", ""))
            return page, "ok", final_title, url, content
        except DisambiguationError as exc:
            if not exc.options:
                return None, "disambiguation_empty", title, "", ""
            fallback = exc.options[0]
            try:
                page = wikipedia.page(fallback, auto_suggest=False)
                final_title = str(getattr(page, "title", fallback))
                url = str(getattr(page, "url", ""))
                content = str(getattr(page, "content", ""))
                return page, "disambiguation_fallback", final_title, url, content
            except Exception:
                return None, "disambiguation_failed", fallback, "", ""
        except PageError:
            return None, "page_not_found", title, "", ""
        except Exception as exc:
            if attempt >= max_attempts:
                print(f"[WARN] Failed to fetch '{title}' after {max_attempts} attempts: {exc}")
                return None, "network_error", title, "", ""
            print(f"[WARN] Transient fetch error for '{title}'. Retrying in {attempt}s...")
            time.sleep(attempt)
    return None, "network_error", title, "", ""


def sample_related_titles(page: object, max_related: int) -> List[str]:
    """精简：利用 tuple 原生支持 startswith 进行高效前缀过滤"""
    if max_related <= 0 or not isinstance(getattr(page, "links", None), list):
        return []

    picked = []
    for name in page.links:
        if isinstance(name, str) and 3 <= len(name) <= 80 and not name.startswith(IGNORE_PREFIXES):
            picked.append(name)
            if len(picked) >= max_related:
                break
    return picked


def write_jsonl(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(rows: List[PageFetchResult], manifest_path: Path) -> None:
    """精简：直接利用 dataclass.asdict() 自动提取字段，无需手动映射"""
    if not rows:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build schema-aligned Turing corpus from Wikipedia")
    parser.add_argument("--titles", nargs="*", help="Optional explicit page titles.")
    parser.add_argument("--filename", default="turing_schema_corpus.jsonl")
    parser.add_argument("--manifest", default="turing_schema_sources.csv")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--source", default="wikipedia_schema")
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-related", type=int, default=12)
    parser.add_argument("--related-depth", type=int, default=2)
    parser.add_argument("--max-pages", type=int, default=30)
    parser.add_argument("--max-paragraphs-per-page", type=int, default=0)
    parser.add_argument("--disable-related", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wikipedia.set_lang(args.lang)
    collected_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 优化：采用 collections.deque 替代 List 作为 BFS 队列，提升 popleft 性能
    page_queue = deque((label, title, title, 0) for label, title in iter_seed_titles(args.titles))
    
    seen_requested: Set[str] = set()
    seen_final_title: Set[str] = set()
    docs: List[Dict[str, str]] = []
    manifest: List[PageFetchResult] = []
    fetched_page_count = 0

    while page_queue:
        seed_label, seed_title, requested_title, depth = page_queue.popleft()
        req_key = requested_title.strip().lower()
        if req_key in seen_requested:
            continue
        seen_requested.add(req_key)

        if 0 < args.max_pages <= fetched_page_count:
            break

        print(f"[INFO] Fetching: {requested_title} (seed={seed_title}, label={seed_label}, depth={depth})")
        page, status, final_title, url, content = safe_fetch_page_and_content(requested_title)
        
        if page is None or not content:
            manifest.append(PageFetchResult(seed_label, seed_title, requested_title, final_title or requested_title, url, status, 0))
            continue

        # 增加爬虫礼貌延迟，防止高并发被 Wikipedia API 封禁
        time.sleep(0.5)

        title_key = final_title.strip().lower()
        if title_key in seen_final_title:
            continue
        seen_final_title.add(title_key)
        fetched_page_count += 1

        paragraphs = split_text_by_paragraph(content, args.min_chars)
        if args.max_paragraphs_per_page > 0:
            paragraphs = paragraphs[:args.max_paragraphs_per_page]

        uid = url or f"wiki:{slugify(final_title)}"
        for idx, paragraph in enumerate(paragraphs, start=1):
            docs.append({
                "doc_id": f"wiki_{slugify(str(final_title))}_{idx:03d}",
                "uid": uid,
                "source": args.source,
                "source_type": "wikipedia",
                "seed_label": seed_label,
                "seed_title": seed_title,
                "title": str(final_title),
                "url": url,
                "lang": args.lang,
                "collected_at": collected_at,
                "text": paragraph,
            })

        manifest.append(PageFetchResult(seed_label, seed_title, requested_title, str(final_title), url, status, len(paragraphs)))

        if not args.disable_related and depth < args.related_depth:
            for related in sample_related_titles(page, args.max_related):
                page_queue.append((seed_label, seed_title, related, depth + 1))

    output_path = RAW_DIR / args.filename
    manifest_path = RAW_DIR / args.manifest
    write_jsonl(docs, output_path)
    write_manifest(manifest, manifest_path)

    print(f"[DONE] Saved {len(docs)} records to {output_path}")
    print(f"[DONE] Saved {len(manifest)} source rows to {manifest_path}")

if __name__ == "__main__":
    main()