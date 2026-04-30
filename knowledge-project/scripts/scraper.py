import argparse
import csv
import json
import re
import time
import warnings
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

try:
    import wikipedia  # type: ignore[import-not-found]
    from wikipedia.exceptions import DisambiguationError, PageError  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'wikipedia'. Install with: pip install wikipedia") from exc

warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

IGNORE_PREFIXES = (
    "Category:",
    "Template:",
    "List of",
    "Wikipedia:",
    "Portal:",
    "Help:",
    "Talk:",
    "Draft:",
    "User:",
)

CORE_SCHEMA_SEEDS: Dict[str, List[str]] = {
    "Person": ["Alan Turing", "Joan Clarke", "Alonzo Church", "Max Newman", "Gordon Welchman", "Robin Gandy"],
    "Organization": ["King's College, Cambridge", "Princeton University", "Bletchley Park", "University of Manchester", "Government Code and Cypher School", "National Physical Laboratory"],
    "Concept": ["Turing machine", "Turing test", "Halting problem", "Church-Turing thesis", "Computability theory", "Morphogenesis"],
    "Artifact": ["Bombe", "Automatic Computing Engine", "Enigma machine", "Pilot ACE"],
    "Publication": ["On Computable Numbers", "Computing Machinery and Intelligence", "The Chemical Basis of Morphogenesis"],
    "Event": ["World War II", "Royal pardon", "Alan Turing law"],
    "Honor": ["Turing Award", "Bank of England 50 note"],
    "Location": ["London", "Maida Vale", "Manchester", "Cambridge", "Princeton, New Jersey"],
}

CORE_TITLE_SET = {title.lower() for titles in CORE_SCHEMA_SEEDS.values() for title in titles}
TURING_KEYWORDS = {
    "alan turing",
    "turing",
    "turing machine",
    "turing test",
    "bletchley park",
    "enigma",
    "computability",
    "cryptanalysis",
    "codebreaking",
    "ace",
    "on computable numbers",
}
SCHEMA_RELATION_HINTS = {
    "born in",
    "died in",
    "graduated from",
    "studied at",
    "worked at",
    "worked for",
    "joined",
    "proposed",
    "designed",
    "developed",
    "wrote",
    "authored",
}
CATEGORY_HINTS = {
    "biography": {"biography", "people", "births", "deaths", "mathematicians", "computer scientists"},
    "computing": {"computing", "computer", "logic", "theory of computation", "artificial intelligence"},
    "cryptography": {"cryptography", "cryptanalysis", "codebreaking", "enigma", "cipher"},
    "history": {"world war ii", "history of computing", "history", "bletchley park"},
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
    layer: str
    page_score: int
    match_reasons: str


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def split_text_by_paragraph(content: str, min_chars: int) -> List[str]:
    paragraphs: List[str] = []
    for block in content.split("\n"):
        text = block.strip()
        if len(text) >= min_chars and not re.match(r"^=+\s*[^=]+\s*=+$", text):
            paragraphs.append(text)
    return paragraphs


def iter_seed_titles(custom_titles: Optional[List[str]]) -> Iterable[Tuple[str, str]]:
    if custom_titles:
        yield from (("Unknown", title) for title in custom_titles)
        return
    for label, titles in CORE_SCHEMA_SEEDS.items():
        for title in titles:
            yield label, title


def safe_fetch_page(title: str) -> Tuple[object | None, str, str, str, str]:
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            page = wikipedia.page(title, auto_suggest=False)
            return page, "ok", str(getattr(page, "title", title)), str(getattr(page, "url", "")), str(getattr(page, "content", ""))
        except DisambiguationError as exc:
            if not exc.options:
                return None, "disambiguation_empty", title, "", ""
            fallback = exc.options[0]
            try:
                page = wikipedia.page(fallback, auto_suggest=False)
                return page, "disambiguation_fallback", str(getattr(page, "title", fallback)), str(getattr(page, "url", "")), str(getattr(page, "content", ""))
            except Exception:
                return None, "disambiguation_failed", fallback, "", ""
        except PageError:
            return None, "page_not_found", title, "", ""
        except Exception as exc:
            if attempt >= max_attempts:
                print(f"[WARN] Failed to fetch '{title}' after {max_attempts} attempts: {exc}")
                return None, "network_error", title, "", ""
            time.sleep(attempt)
    return None, "network_error", title, "", ""


def safe_get_page_links(page: object, max_attempts: int = 3) -> List[str]:
    for attempt in range(1, max_attempts + 1):
        try:
            links = getattr(page, "links", None)
            if isinstance(links, list):
                return [item for item in links if isinstance(item, str)]
            return []
        except Exception as exc:
            if attempt >= max_attempts:
                print(f"[WARN] Failed to load page links after {max_attempts} attempts: {exc}")
                return []
            sleep_seconds = min(2 * attempt, 8)
            print(f"[WARN] Wikipedia links request failed (attempt {attempt}/{max_attempts}), retrying in {sleep_seconds}s...")
            time.sleep(sleep_seconds)
    return []


def safe_get_page_categories(page: object, max_attempts: int = 3) -> List[str]:
    for attempt in range(1, max_attempts + 1):
        try:
            categories = getattr(page, "categories", None)
            if isinstance(categories, list):
                return [str(item).lower() for item in categories if isinstance(item, str)]
            return []
        except Exception as exc:
            if attempt >= max_attempts:
                print(f"[WARN] Failed to load page categories after {max_attempts} attempts: {exc}")
                return []
            sleep_seconds = min(2 * attempt, 8)
            print(f"[WARN] Wikipedia categories request failed (attempt {attempt}/{max_attempts}), retrying in {sleep_seconds}s...")
            time.sleep(sleep_seconds)
    return []


def sample_related_titles(page: object, max_related: int) -> List[str]:
    if max_related <= 0:
        return []
    links = safe_get_page_links(page)
    if not links:
        return []
    picked: List[str] = []
    for name in links:
        if isinstance(name, str) and 3 <= len(name) <= 90 and not name.startswith(IGNORE_PREFIXES):
            picked.append(name)
            if len(picked) >= max_related:
                break
    return picked


def page_categories(page: object) -> List[str]:
    return safe_get_page_categories(page)


def title_hits_seed(title: str) -> bool:
    return title.strip().lower() in CORE_TITLE_SET


def count_keyword_hits(text: str, keywords: Set[str]) -> int:
    lowered = text.lower()
    total = 0
    for keyword in keywords:
        total += len(re.findall(rf"\b{re.escape(keyword)}\b", lowered))
    return total


def count_relation_hint_hits(text: str) -> int:
    lowered = text.lower()
    return sum(1 for hint in SCHEMA_RELATION_HINTS if hint in lowered)


def count_category_hits(categories: List[str]) -> int:
    score = 0
    for category in categories:
        for words in CATEGORY_HINTS.values():
            if any(word in category for word in words):
                score += 1
                break
    return score


def classify_page(
    seed_label: str,
    seed_title: str,
    final_title: str,
    url: str,
    content: str,
    categories: List[str],
    depth: int,
) -> Tuple[str, int, List[str]]:
    reasons: List[str] = []
    score = 0

    final_lower = final_title.lower()
    lead_text = content.split("\n", 1)[0].strip().lower()
    keyword_hits = count_keyword_hits(content, TURING_KEYWORDS)
    relation_hits = count_relation_hint_hits(content)
    category_hits = count_category_hits(categories)
    lead_mentions_center = "alan turing" in lead_text or "alan_turing" in url.lower()
    title_matches_seed = title_hits_seed(final_title)
    direct_relation_page = relation_hits >= 2
    strong_keyword_page = keyword_hits >= 3
    support_keyword_page = keyword_hits >= 1

    if title_matches_seed:
        score += 6
        reasons.append("seed_title_match")
    if seed_title and final_lower == seed_title.lower():
        score += 6
        reasons.append("requested_seed_page")
    if lead_mentions_center:
        score += 6
        reasons.append("lead_mentions_alan_turing")
    if strong_keyword_page:
        score += 4
        reasons.append("strong_keyword_hits")
    elif support_keyword_page:
        score += 2
        reasons.append("keyword_hits")
    if direct_relation_page:
        score += 3
        reasons.append("schema_relation_hints")
    elif relation_hits == 1:
        score += 1
        reasons.append("single_relation_hint")
    if category_hits >= 2:
        score += 3
        reasons.append("topic_categories")
    elif category_hits == 1:
        score += 1
        reasons.append("single_topic_category")
    if depth == 0:
        score += 3
        reasons.append("seed_depth")
    elif depth == 1:
        score += 1
        reasons.append("one_hop_depth")
    if seed_label in {"Person", "Organization", "Concept", "Artifact", "Publication"}:
        score += 1
        reasons.append("schema_seed_label")

    if final_lower == "alan turing":
        return "core", score + 10, reasons + ["center_page"]

    # Core pages now require a hard combination instead of a loose score threshold.
    if title_matches_seed and (lead_mentions_center or direct_relation_page):
        return "core", score, reasons + ["hard_core:title_plus_signal"]
    if lead_mentions_center and direct_relation_page and (strong_keyword_page or category_hits >= 2):
        return "core", score, reasons + ["hard_core:lead_plus_relation"]

    # Support pages can be relevant without satisfying the stronger core gate.
    if title_matches_seed:
        return "support", score, reasons + ["support:title_only"]
    if lead_mentions_center and (support_keyword_page or relation_hits >= 1 or category_hits >= 1):
        return "support", score, reasons + ["support:lead_related"]
    if score >= 6:
        return "support", score, reasons
    return "noise", score, reasons


def build_doc_row(
    final_title: str,
    paragraph: str,
    idx: int,
    uid: str,
    source: str,
    seed_label: str,
    seed_title: str,
    url: str,
    collected_at: str,
    layer: str,
    page_score: int,
    match_reasons: List[str],
    lang: str,
) -> Dict[str, str]:
    return {
        "doc_id": f"wiki_{slugify(final_title)}_{idx:03d}",
        "uid": uid,
        "source": source,
        "source_type": "wikipedia",
        "seed_label": seed_label,
        "seed_title": seed_title,
        "title": final_title,
        "url": url,
        "lang": lang,
        "collected_at": collected_at,
        "layer": layer,
        "page_score": page_score,
        "match_reasons": match_reasons,
        "text": paragraph,
    }


def write_jsonl(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(rows: List[PageFetchResult], manifest_path: Path) -> None:
    if not rows:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Turing-centered, topic-pruned corpus from Wikipedia")
    parser.add_argument("--titles", nargs="*", help="Optional explicit page titles")
    parser.add_argument("--core-output", default="turing_core_corpus.jsonl")
    parser.add_argument("--support-output", default="turing_support_corpus.jsonl")
    parser.add_argument("--noise-output", default="turing_noise_corpus.jsonl")
    parser.add_argument("--combined-output", default="turing_schema_corpus.jsonl")
    parser.add_argument("--manifest", default="turing_schema_sources.csv")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--source", default="wikipedia_turing_pruned")
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-related", type=int, default=12)
    parser.add_argument("--related-depth", type=int, default=2)
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument("--max-paragraphs-per-page", type=int, default=0)
    parser.add_argument("--disable-related", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wikipedia.set_lang(args.lang)
    collected_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    page_queue: Deque[Tuple[str, str, str, int]] = deque(
        (label, title, title, 0) for label, title in iter_seed_titles(args.titles)
    )
    seen_requested: Set[str] = set()
    seen_final_title: Set[str] = set()
    manifest: List[PageFetchResult] = []
    fetched_page_count = 0

    core_docs: List[Dict] = []
    support_docs: List[Dict] = []
    noise_docs: List[Dict] = []

    while page_queue:
        seed_label, seed_title, requested_title, depth = page_queue.popleft()
        req_key = requested_title.strip().lower()
        if req_key in seen_requested:
            continue
        seen_requested.add(req_key)

        if 0 < args.max_pages <= fetched_page_count:
            break

        print(f"[INFO] Fetching: {requested_title} (seed={seed_title}, label={seed_label}, depth={depth})")
        page, status, final_title, url, content = safe_fetch_page(requested_title)
        if page is None or not content:
            manifest.append(PageFetchResult(seed_label, seed_title, requested_title, final_title or requested_title, url, status, 0, "noise", 0, ""))
            continue

        time.sleep(0.5)
        title_key = final_title.strip().lower()
        if title_key in seen_final_title:
            continue
        seen_final_title.add(title_key)
        fetched_page_count += 1

        categories = page_categories(page)
        layer, page_score, reasons = classify_page(
            seed_label=seed_label,
            seed_title=seed_title,
            final_title=final_title,
            url=url,
            content=content,
            categories=categories,
            depth=depth,
        )

        paragraphs = split_text_by_paragraph(content, args.min_chars)
        if args.max_paragraphs_per_page > 0:
            paragraphs = paragraphs[: args.max_paragraphs_per_page]

        uid = url or f"wiki:{slugify(final_title)}"
        target_bucket = core_docs if layer == "core" else support_docs if layer == "support" else noise_docs
        for idx, paragraph in enumerate(paragraphs, start=1):
            target_bucket.append(
                build_doc_row(
                    final_title=final_title,
                    paragraph=paragraph,
                    idx=idx,
                    uid=uid,
                    source=args.source,
                    seed_label=seed_label,
                    seed_title=seed_title,
                    url=url,
                    collected_at=collected_at,
                    layer=layer,
                    page_score=page_score,
                    match_reasons=reasons,
                    lang=args.lang,
                )
            )

        manifest.append(
            PageFetchResult(
                seed_label=seed_label,
                seed_title=seed_title,
                requested_title=requested_title,
                final_title=final_title,
                url=url,
                status=status,
                paragraph_count=len(paragraphs),
                layer=layer,
                page_score=page_score,
                match_reasons="|".join(reasons),
            )
        )

        if not args.disable_related and depth < args.related_depth:
            for related in sample_related_titles(page, args.max_related):
                page_queue.append((seed_label, seed_title, related, depth + 1))

    combined_docs = core_docs + support_docs

    write_jsonl(core_docs, RAW_DIR / args.core_output)
    write_jsonl(support_docs, RAW_DIR / args.support_output)
    write_jsonl(noise_docs, RAW_DIR / args.noise_output)
    write_jsonl(combined_docs, RAW_DIR / args.combined_output)
    write_manifest(manifest, RAW_DIR / args.manifest)

    layer_counts = Counter(row["layer"] for row in combined_docs + noise_docs)
    print(f"[DONE] Core docs: {len(core_docs)} -> {RAW_DIR / args.core_output}")
    print(f"[DONE] Support docs: {len(support_docs)} -> {RAW_DIR / args.support_output}")
    print(f"[DONE] Noise docs: {len(noise_docs)} -> {RAW_DIR / args.noise_output}")
    print(f"[DONE] Combined extraction corpus: {len(combined_docs)} -> {RAW_DIR / args.combined_output}")
    print(f"[DONE] Manifest rows: {len(manifest)} -> {RAW_DIR / args.manifest}")
    print(f"[INFO] Layer counts: {dict(layer_counts)}")


if __name__ == "__main__":
    main()
