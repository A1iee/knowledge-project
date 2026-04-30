import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from neo4j import GraphDatabase


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sanitize_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        if all(item is None or isinstance(item, (bool, int, float, str)) for item in value):
            return value
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def sanitize_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in properties.items():
        cleaned[key] = sanitize_scalar(value)
    return cleaned


def neo4j_safe_label(label: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in label.strip())
    return normalized or "Entity"


def neo4j_safe_rel_type(rel_type: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in rel_type.strip().upper())
    return normalized or "RELATED_TO"


class Neo4jImporter:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def execute(self, query: str, parameters: Dict[str, Any] | None = None) -> None:
        with self.driver.session() as session:
            session.run(query, parameters or {})

    def prepare_db(self) -> None:
        print("[INFO] Preparing Neo4j constraints and indexes...")
        self.execute(
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
        )

    def import_nodes(self, nodes: Iterable[Dict[str, Any]]) -> None:
        print("[INFO] Importing nodes...")
        for node in nodes:
            entity_type = node.get("entity_type") or node.get("label") or "Entity"
            type_label = neo4j_safe_label(str(entity_type))
            attributes = sanitize_properties(node.get("attributes", {}))
            attribute_meta = sanitize_properties(node.get("attribute_meta", {}))

            base_props = sanitize_properties(
                {
                    "id": node.get("id"),
                    "canonical_name": node.get("canonical_name"),
                    "name": node.get("canonical_name"),
                    "entity_type": entity_type,
                    "label": node.get("label"),
                    "aliases": node.get("aliases", []),
                    "source_titles": node.get("source_titles", []),
                    "wiki_url": node.get("wiki_url"),
                    "layer": node.get("layer"),
                    "center_distance": node.get("center_distance"),
                    "centered_on": node.get("centered_on"),
                    "attribute_meta": attribute_meta,
                }
            )

            query = f"""
            MERGE (e:Entity {{id: $id}})
            SET e:{type_label}
            SET e += $base_props
            SET e += $attr_props
            """
            self.execute(
                query,
                {
                    "id": node.get("id"),
                    "base_props": base_props,
                    "attr_props": attributes,
                },
            )

    def import_edges(self, edges: Iterable[Dict[str, Any]]) -> None:
        print("[INFO] Importing edges...")
        for edge in edges:
            rel_type = neo4j_safe_rel_type(str(edge.get("relation", "RELATED_TO")))
            evidence = edge.get("evidence", [])
            sources = edge.get("sources", [])
            source_titles = edge.get("source_titles", [])

            rel_props = sanitize_properties(
                {
                    "relation": edge.get("relation"),
                    "start_name": edge.get("start_name"),
                    "start_label": edge.get("start_label"),
                    "end_name": edge.get("end_name"),
                    "end_label": edge.get("end_label"),
                    "avg_score": edge.get("avg_score"),
                    "final_score": edge.get("final_score"),
                    "weight": edge.get("weight"),
                    "evidence_count": edge.get("evidence_count"),
                    "source_diversity": edge.get("source_diversity"),
                    "layer": edge.get("layer"),
                    "is_core": edge.get("is_core"),
                    "centered_on": edge.get("centered_on"),
                    "decision_sources": edge.get("decision_sources", {}),
                    "evidence": evidence,
                    "evidence_text": "\n".join(evidence) if isinstance(evidence, list) else str(evidence),
                    "sources": sources,
                    "source_titles": source_titles,
                }
            )

            query = f"""
            MATCH (a:Entity {{id: $start_id}})
            MATCH (b:Entity {{id: $end_id}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += $rel_props
            """
            self.execute(
                query,
                {
                    "start_id": edge.get("start_id"),
                    "end_id": edge.get("end_id"),
                    "rel_props": rel_props,
                },
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import fused KG JSONL files into Neo4j.")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="12345678")
    parser.add_argument("--nodes", default="data/output/kg_dev_nodes.jsonl")
    parser.add_argument("--edges", default="data/output/kg_dev_edges.jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)

    if not nodes_path.exists():
        raise SystemExit(f"Nodes file not found: {nodes_path}")
    if not edges_path.exists():
        raise SystemExit(f"Edges file not found: {edges_path}")

    nodes = load_jsonl(nodes_path)
    edges = load_jsonl(edges_path)

    importer = Neo4jImporter(args.uri, args.user, args.password)
    try:
        importer.prepare_db()
        importer.import_nodes(nodes)
        importer.import_edges(edges)
        print(f"[DONE] Imported {len(nodes)} nodes and {len(edges)} edges into Neo4j.")
    finally:
        importer.close()


if __name__ == "__main__":
    main()
