import argparse
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


CENTER_ENTITY_ID = "alan_turing"
CENTER_ENTITY_CANONICAL = "Alan Turing"
CENTER_ENTITY_NAMES = {"alan turing", "alan mathison turing", "turing"}

RELATION_CONSTRAINTS = {
    "BORN_IN": ("Person", "Location"),
    "DIED_IN": ("Person", "Location"),
    "EDUCATED_AT": ("Person", "Organization"),
    "WORKED_AT": ("Person", "Organization"),
    "COLLEAGUE_OF": ("Person", "Person"),
    "PROPOSED": ("Person", "Concept"),
    "WORKED_ON": ("Person", ["Concept", "Artifact"]),
    "AUTHORED": ("Person", "Publication"),
    "PARTICIPATED_IN": (["Person", "Organization"], "Event"),
    "AFFECTED_BY": ("Person", "Event"),
    "HONORED_BY": ("Person", ["Honor", "Organization"]),
    "NAMED_AFTER": (["Honor", "Organization", "Artifact"], "Person"),
}

ALLOWED_NODE_LABELS = {
    "Person",
    "Organization",
    "Location",
    "Concept",
    "Artifact",
    "Event",
    "Publication",
    "Honor",
}

PERSON_ALIAS_RULES = {
    "john von neumann": {"john von neumann", "johann von neumann", "johnny von neumann", "john neumann"},
}

TRASH_ENDINGS = {"und", "and", "the", "of", "for"}


def slugify(text: str) -> str:
    if not text:
        return "unknown"
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def normalize_text(text: str) -> str:
    cleaned = (text or "").strip().strip('"\'. ,;:()[]{}')
    return re.sub(r"\s+", " ", cleaned)


def is_center_entity_text(text: str) -> bool:
    normalized = normalize_text(text).lower()
    return normalized in CENTER_ENTITY_NAMES or slugify(normalized) == CENTER_ENTITY_ID


def canonicalize_person_name(text: str) -> str:
    normalized = normalize_text(text).lower()
    if normalized in CENTER_ENTITY_NAMES:
        return CENTER_ENTITY_CANONICAL
    for canonical, aliases in PERSON_ALIAS_RULES.items():
        if normalized in aliases:
            return canonical.title().replace("Von", "von")
    return normalize_text(text)


def clean_entity_name(text: str, label: str = "") -> Optional[str]:
    normalized = normalize_text(text)
    if len(normalized) < 2:
        return None
    if normalized.lower().split()[-1] in TRASH_ENDINGS:
        return None
    if re.fullmatch(r"[\W_]+", normalized):
        return None
    if normalized.count('"') == 1 or normalized.count("'") == 1 and len(normalized.split()) == 1:
        return None
    if label == "Person":
        normalized = canonicalize_person_name(normalized)
    return normalized


def match_type(real: str, expected) -> bool:
    if isinstance(expected, list):
        return real in expected
    return real == expected


def is_valid_relation_strict(record: Dict) -> bool:
    relation = record.get("relation", "")
    start_label = record.get("start_label", "")
    end_label = record.get("end_label", "")
    if relation not in RELATION_CONSTRAINTS:
        return False
    if start_label not in ALLOWED_NODE_LABELS or end_label not in ALLOWED_NODE_LABELS:
        return False
    exp_start, exp_end = RELATION_CONSTRAINTS[relation]
    return match_type(start_label, exp_start) and match_type(end_label, exp_end)


def schema_validity_bonus(record: Dict) -> float:
    return 0.05 if is_valid_relation_strict(record) else 0.0


def relation_confidence_score(record: Dict) -> float:
    extractor_confidence = float(record.get("confidence", record.get("conf", 0.0)))
    evidence = record.get("evidence", "")
    source = record.get("source", "")
    score = extractor_confidence
    if evidence and len(evidence) > 30:
        score += 0.03
    if source:
        score += 0.02
    if is_center_entity_text(record.get("start_text", "")) or is_center_entity_text(record.get("end_text", "")):
        score += 0.03
    score += schema_validity_bonus(record)
    return round(score, 6)


def compute_center_distance(edges: Iterable[Tuple[str, str, str]]) -> Dict[str, int]:
    graph: Dict[str, Set[str]] = defaultdict(set)
    for start_id, _, end_id in edges:
        graph[start_id].add(end_id)
        graph[end_id].add(start_id)

    if CENTER_ENTITY_ID not in graph:
        return {}

    distance = {CENTER_ENTITY_ID: 0}
    queue = deque([CENTER_ENTITY_ID])
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if neighbor in distance:
                continue
            distance[neighbor] = distance[current] + 1
            queue.append(neighbor)
    return distance


def infer_layer(start_id: str, end_id: str, distance: Dict[str, int]) -> Optional[str]:
    if start_id == CENTER_ENTITY_ID or end_id == CENTER_ENTITY_ID:
        return "core"
    start_dist = distance.get(start_id)
    end_dist = distance.get(end_id)
    if start_dist is None or end_dist is None:
        return None
    return "core" if max(start_dist, end_dist) <= 2 else "expanded"


def entity_key_from_relation(text: str, label: str, source: str = "") -> str:
    cleaned = clean_entity_name(text, label)
    if not cleaned:
        return "unknown"
    if is_center_entity_text(cleaned):
        return CENTER_ENTITY_ID
    if source and "wikipedia.org/wiki/" in source and is_center_entity_text(cleaned):
        return CENTER_ENTITY_ID
    return slugify(cleaned)


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def aggregate_entity_metadata(
    entity_store: Dict[str, Dict],
    entity_id: str,
    name: str,
    label: str,
    source_title: str = "",
    source_url: str = "",
) -> None:
    if entity_id not in entity_store:
        entity_store[entity_id] = {
            "canonical_name": name,
            "aliases": set(),
            "entity_type_counter": Counter(),
            "source_titles": set(),
            "wiki_url": source_url if "wikipedia.org/wiki/" in source_url else "",
        }
    meta = entity_store[entity_id]
    meta["aliases"].add(name)
    meta["entity_type_counter"][label] += 1
    if len(name) > len(meta["canonical_name"]):
        meta["canonical_name"] = name
    if source_title:
        meta["source_titles"].add(source_title)
    if not meta["wiki_url"] and "wikipedia.org/wiki/" in source_url:
        meta["wiki_url"] = source_url


def judge_attribute_value(
    entity_label: str,
    attr_name: str,
    attr_value,
    confidence: float,
    source_title: str,
    source_url: str,
    evidence: str,
) -> float:
    score = confidence
    if entity_label == "Person" and attr_name in {"birth_date", "death_date", "nationality", "occupation"}:
        score += 0.05
    if entity_label == "Organization" and attr_name in {"established_year", "location", "org_type"}:
        score += 0.05
    if entity_label == "Publication" and attr_name in {"publish_year", "venue"}:
        score += 0.05
    if entity_label in {"Artifact", "Concept"} and attr_name in {"creation_year", "description"}:
        score += 0.05
    if source_title and source_title == normalize_text(source_title):
        score += 0.01
    if source_url and "wikipedia.org/wiki/" in source_url:
        score += 0.02
    if evidence and len(evidence) > 40:
        score += 0.02
    return round(score, 6)


def choose_best_entity_type(counter: Counter) -> str:
    if not counter:
        return "Entity"
    return counter.most_common(1)[0][0]


def fuse_knowledge(rel_file: str, attr_file: str, output_rel: str, output_ent: str, threshold: float = 0.75) -> None:
    relation_path = Path(rel_file)
    attr_path = Path(attr_file)
    output_rel_path = Path(output_rel)
    output_ent_path = Path(output_ent)

    entity_store: Dict[str, Dict] = {}
    relation_candidates: Dict[Tuple[str, str, str], Dict] = defaultdict(
        lambda: {
            "evidence": [],
            "sources": set(),
            "source_titles": set(),
            "score_sum": 0.0,
            "score_max": 0.0,
            "count": 0,
            "start_label": "",
            "end_label": "",
            "decision_sources": Counter(),
        }
    )
    attribute_candidates: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    edge_keys: List[Tuple[str, str, str]] = []

    for record in read_jsonl(relation_path):
        start_label = record.get("start_label", "")
        end_label = record.get("end_label", "")
        start_text = clean_entity_name(record.get("start_text", ""), start_label)
        end_text = clean_entity_name(record.get("end_text", ""), end_label)
        if not start_text or not end_text:
            continue
        if slugify(start_text) == slugify(end_text):
            continue

        record["start_text"] = start_text
        record["end_text"] = end_text
        if not is_valid_relation_strict(record):
            continue

        score = relation_confidence_score(record)
        if score < threshold:
            continue

        start_id = entity_key_from_relation(start_text, start_label, record.get("source", ""))
        end_id = entity_key_from_relation(end_text, end_label, record.get("source", ""))
        if start_id == end_id:
            continue

        aggregate_entity_metadata(
            entity_store,
            start_id,
            start_text,
            start_label,
            source_title=normalize_text(record.get("doc_id", "")),
            source_url=record.get("source", ""),
        )
        aggregate_entity_metadata(
            entity_store,
            end_id,
            end_text,
            end_label,
            source_title=normalize_text(record.get("doc_id", "")),
            source_url=record.get("source", ""),
        )

        relation = record["relation"]
        key = (start_id, relation, end_id)
        bucket = relation_candidates[key]
        bucket["evidence"].append(record.get("evidence", ""))
        bucket["sources"].add(record.get("source", ""))
        bucket["source_titles"].add(record.get("doc_id", ""))
        bucket["score_sum"] += score
        bucket["score_max"] = max(bucket["score_max"], score)
        bucket["count"] += 1
        bucket["start_label"] = start_label
        bucket["end_label"] = end_label
        bucket["decision_sources"][record.get("decision_source", "unknown")] += 1
        edge_keys.append(key)

    for record in read_jsonl(attr_path):
        entity_label = record.get("entity_label", "")
        entity_text = clean_entity_name(record.get("entity_text", record.get("entity", "")), entity_label)
        attr_name = record.get("attribute_name", record.get("attr", ""))
        attr_value = record.get("attribute_value", record.get("value", ""))
        confidence = float(record.get("confidence", record.get("conf", 0.0)))
        if confidence < threshold:
            continue
        if not entity_text or not attr_name:
            continue

        entity_id = entity_key_from_relation(entity_text, entity_label, record.get("source", ""))
        aggregate_entity_metadata(
            entity_store,
            entity_id,
            entity_text,
            entity_label if entity_label in ALLOWED_NODE_LABELS else "Entity",
            source_title=normalize_text(record.get("doc_id", "")),
            source_url=record.get("source", ""),
        )

        score = judge_attribute_value(
            entity_label=entity_label,
            attr_name=attr_name,
            attr_value=attr_value,
            confidence=confidence,
            source_title=record.get("doc_id", ""),
            source_url=record.get("source", ""),
            evidence=record.get("evidence", ""),
        )
        attribute_candidates[entity_id][attr_name].append(
            {
                "value": attr_value,
                "score": score,
                "confidence": confidence,
                "source": record.get("source", ""),
                "doc_id": record.get("doc_id", ""),
                "evidence": record.get("evidence", ""),
                "entity_label": entity_label,
            }
        )

    distance = compute_center_distance(edge_keys)

    final_relations: List[Dict] = []
    for (start_id, relation, end_id), info in relation_candidates.items():
        layer = infer_layer(start_id, end_id, distance)
        if layer is None:
            continue

        evidence_count = info["count"]
        source_diversity = len([source for source in info["sources"] if source])
        final_score = round(
            (info["score_sum"] / max(evidence_count, 1))
            + min(evidence_count, 5) * 0.02
            + min(source_diversity, 3) * 0.03,
            6,
        )

        final_relations.append(
            {
                "start_id": start_id,
                "start_name": entity_store[start_id]["canonical_name"],
                "start_label": choose_best_entity_type(entity_store[start_id]["entity_type_counter"]),
                "relation": relation,
                "end_id": end_id,
                "end_name": entity_store[end_id]["canonical_name"],
                "end_label": choose_best_entity_type(entity_store[end_id]["entity_type_counter"]),
                "avg_score": round(info["score_sum"] / max(evidence_count, 1), 4),
                "final_score": final_score,
                "weight": evidence_count,
                "evidence_count": evidence_count,
                "source_diversity": source_diversity,
                "layer": layer,
                "is_core": layer == "core",
                "centered_on": CENTER_ENTITY_ID,
                "decision_sources": dict(info["decision_sources"]),
                "evidence": list(dict.fromkeys(e for e in info["evidence"] if e))[:5],
                "sources": sorted(source for source in info["sources"] if source),
                "source_titles": sorted(title for title in info["source_titles"] if title),
            }
        )

    used_entity_ids: Set[str] = set()
    for relation in final_relations:
        used_entity_ids.add(relation["start_id"])
        used_entity_ids.add(relation["end_id"])

    final_entities: List[Dict] = []
    for entity_id in sorted(used_entity_ids):
        meta = entity_store.get(entity_id, {})
        attr_values: Dict[str, Dict] = {}
        for attr_name, candidates in attribute_candidates.get(entity_id, {}).items():
            best = sorted(
                candidates,
                key=lambda item: (
                    -item["score"],
                    -(1 if item["source"] and "wikipedia.org/wiki/" in item["source"] else 0),
                    -len(item["evidence"]),
                ),
            )[0]
            attr_values[attr_name] = {
                "value": best["value"],
                "confidence": round(best["confidence"], 4),
                "source": best["source"],
                "doc_id": best["doc_id"],
            }

        entity_type = choose_best_entity_type(meta.get("entity_type_counter", Counter()))
        dist = distance.get(entity_id)
        layer = "core" if dist is not None and dist <= 2 else "expanded"

        final_entities.append(
            {
                "id": entity_id,
                "canonical_name": meta.get("canonical_name", entity_id),
                "aliases": sorted(meta.get("aliases", set())),
                "entity_type": entity_type,
                "label": entity_type,
                "source_titles": sorted(meta.get("source_titles", set())),
                "wiki_url": meta.get("wiki_url", ""),
                "layer": layer,
                "center_distance": dist,
                "centered_on": CENTER_ENTITY_ID,
                "attributes": {name: payload["value"] for name, payload in attr_values.items()},
                "attribute_meta": attr_values,
            }
        )

    output_rel_path.parent.mkdir(parents=True, exist_ok=True)
    output_ent_path.parent.mkdir(parents=True, exist_ok=True)

    with output_rel_path.open("w", encoding="utf-8") as f:
        for relation in final_relations:
            f.write(json.dumps(relation, ensure_ascii=False) + "\n")

    with output_ent_path.open("w", encoding="utf-8") as f:
        for entity in final_entities:
            f.write(json.dumps(entity, ensure_ascii=False) + "\n")

    print("[DONE] Knowledge fusion completed")
    print(f"Relations: {len(final_relations)}")
    print(f"Entities: {len(final_entities)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse extracted relations and attributes into a centered Turing KG.")
    parser.add_argument("--rel-file", default="data/output/relations_dev.jsonl")
    parser.add_argument("--attr-file", default="data/output/attributes_dev.jsonl")
    parser.add_argument("--output-rel", default="data/output/kg_dev_edges.jsonl")
    parser.add_argument("--output-ent", default="data/output/kg_dev_nodes.jsonl")
    parser.add_argument("--threshold", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fuse_knowledge(
        rel_file=args.rel_file,
        attr_file=args.attr_file,
        output_rel=args.output_rel,
        output_ent=args.output_ent,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
