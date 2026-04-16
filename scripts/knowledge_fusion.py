from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ==========================================
# 1. 统一本体定义层 (Ontology Schema)
# ==========================================

ONTOLOGY = {
    "attributes": {
        "birth_date": {"cardinality": "single", "type": "date"},
        "death_date": {"cardinality": "single", "type": "date"},
        "established_year": {"cardinality": "single", "type": "date"},
        "occupation": {"cardinality": "multi", "type": "string"},
        "aliases": {"cardinality": "multi", "type": "string"},
        "nationality": {"cardinality": "multi", "type": "string"},
        "org_type": {"cardinality": "single", "type": "string"}
    },
    "relations": {
        "BORN_IN": {"domain": ["Person"], "range": ["Location"], "cardinality": "single"},
        "DIED_IN": {"domain": ["Person"], "range": ["Location"], "cardinality": "single"},
        "EDUCATED_AT": {"domain": ["Person"], "range": ["Organization"], "cardinality": "multi"},
        "WORKED_AT": {"domain": ["Person"], "range": ["Organization"], "cardinality": "multi"},
        "COLLEAGUE_OF": {"domain": ["Person"], "range": ["Person"], "cardinality": "multi"},
        "FRIEND_OF": {"domain": ["Person"], "range": ["Person"], "cardinality": "multi"},
        
        "PROPOSED": {"domain": ["Person"], "range": ["Concept"], "cardinality": "multi"},
        "INVENTED": {"domain": ["Person"], "range": ["Artifact"], "cardinality": "multi"},
        "AUTHORED": {"domain": ["Person"], "range": ["Publication"], "cardinality": "multi"},
        "WORKED_ON": {"domain": ["Person"], "range": ["Artifact", "Concept"], "cardinality": "multi"},
        
        "PARTICIPATED_IN": {"domain": ["Person", "Organization"], "range": ["Event"], "cardinality": "multi"},
        "AFFECTED_BY": {"domain": ["Person"], "range": ["Event"], "cardinality": "multi"},
        "AWARDED": {"domain": ["Person"], "range": ["Honor"], "cardinality": "multi"},
        "NAMED_AFTER": {"domain": ["Honor", "Organization"], "range": ["Person"], "cardinality": "single"}
    }
}

# ==========================================
# 2. 冲突解决引擎与数据结构
# ==========================================

def get_source_weight(source: str, confidence: str) -> float:
    """计算数据来源的信任权重 (支持拓展)"""
    src_lower = str(source).lower()
    
    # 1. 来源基准分数
    if "wikidata" in src_lower:
        base_score = 10.0  # 结构化知识库最高信任
    elif "dbpedia" in src_lower:
        base_score = 9.0
    elif "wikipedia" in src_lower or "wiki" in src_lower:
        base_score = 8.0  # 半结构化百科较高信任
    elif "expert" in src_lower:
        base_score = 15.0 # 专家人工录入绝对信任
    else:
        base_score = 5.0  # 普通 NLP 规则抽取
        
    # 2. 抽取置信度加成
    conf_lower = str(confidence).lower()
    if conf_lower == "high":
        multiplier = 1.2
    elif conf_lower == "low":
        multiplier = 0.5
    else:
        multiplier = 1.0
        
    return base_score * multiplier


@dataclass
class FactEvidence:
    value: str
    source: str
    confidence: str
    weight: float


def resolve_conflict(evidences: List[FactEvidence], cardinality: str) -> Tuple[List[str], List[str]]:
    """
    核心冲突解决算法
    返回: (保留的终值列表, 融合的来源集合列表)
    """
    if not evidences:
        return [], []
        
    score_board: Dict[str, float] = defaultdict(float)
    sources_board: Dict[str, Set[str]] = defaultdict(set)
    
    for ev in evidences:
        norm_val = ev.value.strip()
        if not norm_val:
            continue
        score_board[norm_val] += ev.weight
        sources_board[norm_val].add(ev.source)

    if not score_board:
        return [], []

    if cardinality == "single":
        best_val = max(score_board.items(), key=lambda x: x[1])[0]
        return [best_val], ["|".join(sorted(sources_board[best_val]))]
        
    final_vals, final_sources = [], []
    threshold = 3.0 
    for val, score in score_board.items():
        if score >= threshold:
            final_vals.append(val)
            final_sources.append("|".join(sorted(sources_board[val])))
    
    return final_vals, final_sources

# ==========================================
# 3. 主融合流水线
# ==========================================

def load_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser(description="Knowledge Fusion & Ontology Alignment")
    p.add_argument("--nodes-in", default="data/output/kg_final/nodes.csv")
    p.add_argument("--edges-in", default="data/output/kg_final/edges_normalized.csv")
    p.add_argument("--attrs-in", default="data/output/kg_final/attrs_normalized.csv")
    p.add_argument("--edges-out", default="data/output/kg_master/master_edges.csv")
    p.add_argument("--attrs-out", default="data/output/kg_master/master_attrs.csv")
    args = p.parse_args()

    Path(args.edges_out).parent.mkdir(parents=True, exist_ok=True)

    print("Loading nodes...")
    nodes_data = load_csv(Path(args.nodes_in))
    node_labels: Dict[str, str] = {row["id"]: row["label"] for row in nodes_data}

    # 属性融合
    print("Fusing Attributes...")
    raw_attrs = load_csv(Path(args.attrs_in))
    attr_groups: Dict[str, Dict[str, List[FactEvidence]]] = defaultdict(lambda: defaultdict(list))
    
    for row in raw_attrs:
        uid, attr_name, val = row["entity_uid"], row["attribute_name"], row["attribute_value"]
        if attr_name not in ONTOLOGY["attributes"]:
            continue
            
        weight = get_source_weight(row.get("source", ""), row.get("confidence", "high"))
        attr_groups[uid][attr_name].append(FactEvidence(val, row.get("source", ""), row.get("confidence", ""), weight))

    master_attrs = []
    for uid, attrs_dict in attr_groups.items():
        for attr_name, evidences in attrs_dict.items():
            cardinality = ONTOLOGY["attributes"][attr_name]["cardinality"]
            resolved_vals, merged_sources = resolve_conflict(evidences, cardinality)
            
            for v, s in zip(resolved_vals, merged_sources):
                master_attrs.append({
                    "entity_uid": uid, "attribute_name": attr_name, "attribute_value": v,
                    "fused_sources": s, "evidence_count": len(evidences)
                })

    # 关系融合
    print("Fusing Relations...")
    raw_edges = load_csv(Path(args.edges_in))
    edge_groups: Dict[str, Dict[str, List[FactEvidence]]] = defaultdict(lambda: defaultdict(list))
    dropped_by_schema = 0
    
    for row in raw_edges:
        start_uid, end_uid, rel = row["start_uid"], row["end_uid"], row["relation"]
        if rel not in ONTOLOGY["relations"]:
            dropped_by_schema += 1
            continue
            
        rel_schema = ONTOLOGY["relations"][rel]
        start_label, end_label = node_labels.get(start_uid), node_labels.get(end_uid)
        
        if start_label not in rel_schema["domain"] or end_label not in rel_schema["range"]:
            dropped_by_schema += 1
            continue
            
        weight = get_source_weight(row.get("source", ""), row.get("confidence", "high"))
        edge_groups[start_uid][rel].append(FactEvidence(end_uid, row.get("source", ""), row.get("confidence", ""), weight))

    master_edges = []
    for start_uid, rel_dict in edge_groups.items():
        for rel_name, evidences in rel_dict.items():
            cardinality = ONTOLOGY["relations"][rel_name]["cardinality"]
            resolved_ends, merged_sources = resolve_conflict(evidences, cardinality)
            
            for end_uid, s in zip(resolved_ends, merged_sources):
                master_edges.append({
                    "start_uid": start_uid, "relation": rel_name, "end_uid": end_uid,
                    "fused_sources": s, "evidence_count": len(evidences)
                })

    print(f"Dropped {dropped_by_schema} edges due to Schema violations.")
    print("Exporting Master Knowledge Graph...")
    
    with open(args.attrs_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["entity_uid", "attribute_name", "attribute_value", "fused_sources", "evidence_count"])
        writer.writeheader()
        writer.writerows(master_attrs)

    with open(args.edges_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["start_uid", "relation", "end_uid", "fused_sources", "evidence_count"])
        writer.writeheader()
        writer.writerows(master_edges)

    print(f"Master Attributes: {len(master_attrs)} rows")
    print(f"Master Edges:      {len(master_edges)} rows")


if __name__ == "__main__":
    main()