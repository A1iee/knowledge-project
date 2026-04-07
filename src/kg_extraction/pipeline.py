import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .rules import (
    ALIAS_MAP,
    ENTITY_LABELS,
    GAZETTEER,
    LABEL_HINT_PATTERNS,
    RELATION_PATTERNS,
    RELATION_SCHEMA,
)


YEAR_PATTERN = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
CAPITALIZED_NGRAM_PATTERN = re.compile(r"\b([A-Z][A-Za-z\-']+(?:\s+[A-Z][A-Za-z\-']+){0,4})\b")


@dataclass
class Mention:
    mention_text: str
    canonical_name: str
    label: str
    doc_id: str
    sentence_id: int
    context: str
    source: str
    evidence: str
    confidence: str


@dataclass
class Entity:
    entity_id: str
    name: str
    label: str
    aliases: set = field(default_factory=set)
    sources: set = field(default_factory=set)
    evidences: set = field(default_factory=set)
    years: set = field(default_factory=set)
    org_hints: set = field(default_factory=set)
    pending_review: bool = False
    confidence: str = "medium"


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_name(name: str) -> str:
    name = name.strip().strip(".,;:()[]{}")
    if name in ALIAS_MAP:
        return ALIAS_MAP[name]
    return re.sub(r"\s+", " ", name)


def split_sentences(text: str) -> List[str]:
    parts = SENTENCE_SPLIT_PATTERN.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def load_corpus(jsonl_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def detect_label_by_hints(mention: str, sentence: str) -> str:
    for label, patterns in LABEL_HINT_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(mention) or pattern.search(sentence):
                return label
    return "Person"


def extract_candidate_mentions(sentence: str) -> List[Tuple[str, str, str]]:
    found: List[Tuple[str, str, str]] = []

    for label, names in GAZETTEER.items():
        for name in names:
            if re.search(rf"\b{re.escape(name)}\b", sentence):
                found.append((name, normalize_name(name), label))

    for raw in CAPITALIZED_NGRAM_PATTERN.findall(sentence):
        if len(raw) < 3:
            continue
        canonical = normalize_name(raw)
        if any(canonical == item[1] for item in found):
            continue
        label = detect_label_by_hints(raw, sentence)
        if label in ENTITY_LABELS:
            found.append((raw, canonical, label))

    return found


def extract_mentions(corpus_rows: Iterable[Dict[str, str]]) -> List[Mention]:
    mentions: List[Mention] = []
    for row in corpus_rows:
        doc_id = row.get("doc_id", "unknown_doc")
        source = row.get("source", "unknown_source")
        text = row.get("text", "")
        sentences = split_sentences(text)
        for sid, sentence in enumerate(sentences, start=1):
            for mention_text, canonical, label in extract_candidate_mentions(sentence):
                confidence = "high" if mention_text in GAZETTEER.get(label, set()) else "medium"
                mentions.append(
                    Mention(
                        mention_text=mention_text,
                        canonical_name=canonical,
                        label=label,
                        doc_id=doc_id,
                        sentence_id=sid,
                        context=sentence,
                        source=source,
                        evidence=sentence,
                        confidence=confidence,
                    )
                )
    return mentions


def disambiguation_score(entity: Entity, mention: Mention) -> float:
    score = 0.0

    if entity.name == mention.canonical_name:
        score += 0.7
    elif mention.canonical_name in entity.aliases:
        score += 0.5

    if entity.label == mention.label:
        score += 0.2

    years = set(YEAR_PATTERN.findall(mention.context))
    if years and entity.years and years.intersection(entity.years):
        score += 0.2

    for org in GAZETTEER.get("Organization", set()):
        if org in mention.context and org in entity.org_hints:
            score += 0.2
            break

    return score


def build_entity_id(label: str, canonical_name: str, seq: int = 0) -> str:
    base = f"{label}:{slugify(canonical_name)}"
    return base if seq == 0 else f"{base}_{seq}"


def resolve_entities(mentions: List[Mention]) -> Tuple[List[Entity], Dict[Tuple[str, str, int], str]]:
    entities: List[Entity] = []
    mention_entity_map: Dict[Tuple[str, str, int], str] = {}

    for mention in mentions:
        best_idx = -1
        best_score = -1.0

        for idx, entity in enumerate(entities):
            if entity.label != mention.label:
                continue
            score = disambiguation_score(entity, mention)
            if score > best_score:
                best_idx = idx
                best_score = score

        if best_idx >= 0 and best_score >= 0.8:
            entity = entities[best_idx]
            entity.aliases.add(mention.mention_text)
            entity.sources.add(mention.source)
            entity.evidences.add(mention.evidence)
            entity.years.update(YEAR_PATTERN.findall(mention.context))
            for org in GAZETTEER.get("Organization", set()):
                if org in mention.context:
                    entity.org_hints.add(org)
            if best_score >= 1.1:
                entity.confidence = "high"
            mention_entity_map[(mention.doc_id, mention.canonical_name, mention.sentence_id)] = entity.entity_id
            continue

        same_name_count = sum(
            1
            for e in entities
            if e.label == mention.label and e.name == mention.canonical_name
        )
        entity_id = build_entity_id(mention.label, mention.canonical_name, seq=same_name_count)
        new_entity = Entity(
            entity_id=entity_id,
            name=mention.canonical_name,
            label=mention.label,
            aliases={mention.mention_text},
            sources={mention.source},
            evidences={mention.evidence},
            years=set(YEAR_PATTERN.findall(mention.context)),
            confidence=mention.confidence,
            pending_review=mention.confidence == "low",
        )
        for org in GAZETTEER.get("Organization", set()):
            if org in mention.context:
                new_entity.org_hints.add(org)
        entities.append(new_entity)
        mention_entity_map[(mention.doc_id, mention.canonical_name, mention.sentence_id)] = entity_id

    return entities, mention_entity_map


def relation_candidates_for_sentence(
    sentence: str,
    mentions_in_sentence: List[Mention],
    mention_entity_map: Dict[Tuple[str, str, int], str],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    lowered = sentence.lower()

    for relation, patterns in RELATION_PATTERNS.items():
        if not any(pattern.search(lowered) for pattern in patterns):
            continue

        expected_start, expected_end = RELATION_SCHEMA[relation]
        for m_start in mentions_in_sentence:
            for m_end in mentions_in_sentence:
                if m_start is m_end:
                    continue

                if expected_start != "Any" and m_start.label != expected_start:
                    continue
                if expected_end != "Any" and m_end.label != expected_end:
                    continue

                start_id = mention_entity_map.get((m_start.doc_id, m_start.canonical_name, m_start.sentence_id))
                end_id = mention_entity_map.get((m_end.doc_id, m_end.canonical_name, m_end.sentence_id))
                if not start_id or not end_id or start_id == end_id:
                    continue

                confidence = "high" if relation != "RELATED_TO" else "medium"

                rows.append(
                    {
                        "start_id": start_id,
                        "end_id": end_id,
                        "relation": relation,
                        "evidence": sentence,
                        "source": m_start.source,
                        "confidence": confidence,
                        "extract_method": "rule",
                        "disputed": "false",
                    }
                )

    return rows


def extract_relations(
    mentions: List[Mention],
    mention_entity_map: Dict[Tuple[str, str, int], str],
) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, int], List[Mention]] = {}
    for mention in mentions:
        grouped.setdefault((mention.doc_id, mention.sentence_id), []).append(mention)

    relations: List[Dict[str, str]] = []
    for (_, _), sentence_mentions in grouped.items():
        sentence = sentence_mentions[0].context
        relations.extend(relation_candidates_for_sentence(sentence, sentence_mentions, mention_entity_map))

    dedup: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in relations:
        key = (row["start_id"], row["relation"], row["end_id"])
        if key not in dedup:
            dedup[key] = row
            continue

        existing = dedup[key]
        if row["evidence"] not in existing["evidence"]:
            existing["evidence"] = existing["evidence"] + " | " + row["evidence"]
        if existing["confidence"] == "medium" and row["confidence"] == "high":
            existing["confidence"] = "high"

    return list(dedup.values())


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_pipeline(repo_root: Path, raw_path: Path | None = None) -> None:
    if raw_path is None:
        default_raw = repo_root / "data" / "raw" / "turing_corpus.jsonl"
        fallback_raw = repo_root.parent / "data" / "raw" / "turing_corpus.jsonl"
        if default_raw.exists():
            raw_path = default_raw
        elif fallback_raw.exists():
            raw_path = fallback_raw
        else:
            raise FileNotFoundError(
                "Corpus file not found. Expected one of: "
                f"{default_raw} or {fallback_raw}. "
                "Generate corpus with: python scripts/build_wikipedia_corpus.py"
            )

    intermediate_dir = repo_root / "data" / "intermediate"
    output_dir = repo_root / "data" / "output"

    corpus = load_corpus(raw_path)
    mentions = extract_mentions(corpus)
    entities, mention_entity_map = resolve_entities(mentions)
    relations = extract_relations(mentions, mention_entity_map)

    mention_rows = [
        {
            "doc_id": m.doc_id,
            "sentence_id": str(m.sentence_id),
            "mention_text": m.mention_text,
            "canonical_name": m.canonical_name,
            "label": m.label,
            "context": m.context,
            "source": m.source,
            "evidence": m.evidence,
            "confidence": m.confidence,
        }
        for m in mentions
    ]

    entity_rows = [
        {
            "entity_id": e.entity_id,
            "name": e.name,
            "label": e.label,
            "aliases": "|".join(sorted(e.aliases)),
            "source": "|".join(sorted(e.sources)),
            "evidence": " | ".join(sorted(e.evidences)),
            "confidence": e.confidence,
            "pending_review": "true" if e.pending_review else "false",
        }
        for e in entities
    ]

    edge_rows = [r for r in relations if r["confidence"] in {"medium", "high"}]

    write_csv(
        intermediate_dir / "entity_mentions.csv",
        [
            "doc_id",
            "sentence_id",
            "mention_text",
            "canonical_name",
            "label",
            "context",
            "source",
            "evidence",
            "confidence",
        ],
        mention_rows,
    )

    write_csv(
        intermediate_dir / "entities_resolved.csv",
        [
            "entity_id",
            "name",
            "label",
            "aliases",
            "source",
            "evidence",
            "confidence",
            "pending_review",
        ],
        entity_rows,
    )

    write_csv(
        intermediate_dir / "relation_candidates.csv",
        [
            "start_id",
            "end_id",
            "relation",
            "evidence",
            "source",
            "confidence",
            "extract_method",
            "disputed",
        ],
        relations,
    )

    write_csv(
        output_dir / "nodes.csv",
        ["entity_id", "name", "label", "aliases", "source"],
        [
            {
                "entity_id": e.entity_id,
                "name": e.name,
                "label": e.label,
                "aliases": "|".join(sorted(e.aliases)),
                "source": "|".join(sorted(e.sources)),
            }
            for e in entities
        ],
    )

    write_csv(
        output_dir / "edges.csv",
        ["start_id", "end_id", "relation", "evidence", "source", "confidence"],
        [
            {
                "start_id": row["start_id"],
                "end_id": row["end_id"],
                "relation": row["relation"],
                "evidence": row["evidence"],
                "source": row["source"],
                "confidence": row["confidence"],
            }
            for row in edge_rows
        ],
    )
