from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional

# ==========================================
# 1. 领域特定词典与正则规则
# ==========================================

# 预定义的硬编码缩写映射 (针对图灵/计算机科学领域)
HARDCODED_ACRONYMS = {
    "mit": "massachusetts_institute_of_technology",
    "npl": "national_physical_laboratory",
    "gchq": "government_communications_headquarters",
    "gc_cs": "government_code_and_cypher_school",
    "ias": "institute_for_advanced_study",
    "acm": "association_for_computing_machinery",
    "rs": "royal_society",
    "ucl": "university_college_london",
    "ibm": "international_business_machines",
    "lse": "london_school_of_economics"
}

# 清理无用前缀/后缀的正则
_CLEAN_PREFIX_RE = re.compile(r"^(the|a|an)\s+", flags=re.IGNORECASE)
_ORG_UNIV_RE = re.compile(r"^university of (.*)$", flags=re.IGNORECASE)

def clean_entity_text(text: str, label: str) -> str:
    """清理实体文本的表面干扰"""
    s = text.strip()
    # 移除冠词
    s = _CLEAN_PREFIX_RE.sub("", s)
    
    # 机构名标准化: "University of Cambridge" -> "Cambridge University"
    if label == "Organization":
        m = _ORG_UNIV_RE.match(s)
        if m:
            s = f"{m.group(1)} University"
    
    return s.strip()

def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.strip().lower()).strip('_')

# ==========================================
# 2. 实体数据结构
# ==========================================

@dataclass
class RawEntity:
    uid: str
    label: str
    text: str
    frequency: int = 1

@dataclass
class CanonicalNode:
    uid: str
    label: str
    primary_name: str
    aliases: Set[str] = field(default_factory=set)

# ==========================================
# 3. 核心聚类与消歧算法
# ==========================================

def is_initials_match(short_name: str, long_name: str) -> bool:
    """
    判断是否为缩写匹配，例如 "A. Turing" 匹配 "Alan Turing", "A. M. Turing" 匹配 "Alan Mathison Turing"
    """
    short_parts = [p.strip() for p in re.split(r'[\s.]+', short_name) if p.strip()]
    long_parts = [p.strip() for p in re.split(r'[\s.]+', long_name) if p.strip()]
    
    if len(short_parts) > len(long_parts) or not short_parts:
        return False
        
    # 检查姓氏是否一致 (最后一个词)
    if short_parts[-1].lower() != long_parts[-1].lower():
        return False
        
    # 检查前面的缩写字母
    for sp, lp in zip(short_parts[:-1], long_parts[:-1]):
        if len(sp) == 1: # 是首字母
            if sp[0].lower() != lp[0].lower():
                return False
        else: # 可能是全名匹配
            if sp.lower() != lp.lower():
                return False
    return True

def resolve_entities(raw_entities: List[RawEntity]) -> Tuple[List[CanonicalNode], Dict[str, str]]:
    """
    实体消歧：将表面不同的实体合并到规范化实体中
    返回: (规范化节点列表, 原始UID到规范UID的映射表)
    """
    # 按 Label 分组处理 (绝不跨 Label 合并)
    grouped: Dict[str, List[RawEntity]] = defaultdict(list)
    for ent in raw_entities:
        grouped[ent.label].append(ent)

    mapping: Dict[str, str] = {}
    nodes_dict: Dict[str, CanonicalNode] = {}

    for label, ents in grouped.items():
        # 预处理清理
        for e in ents:
            e.text = clean_entity_text(e.text, label)
            
        # 启发式聚类：将名称长的作为主实体候选，名称短的往往是简称
        # 按照 (文本长度降序, 频率降序) 排序
        ents_sorted = sorted(ents, key=lambda x: (len(x.text), x.frequency), reverse=True)
        
        canonical_candidates: List[CanonicalNode] = []
        
        for raw in ents_sorted:
            matched_canonical = None
            raw_slug = slugify(raw.text)
            
            # 1. 检查硬编码词典 (针对机构)
            if label == "Organization" and raw_slug in HARDCODED_ACRONYMS:
                target_slug = HARDCODED_ACRONYMS[raw_slug]
                for cand in canonical_candidates:
                    if slugify(cand.primary_name) == target_slug:
                        matched_canonical = cand
                        break

            # 2. 遍历已有的权威实体候选，寻找匹配项
            if not matched_canonical:
                for cand in canonical_candidates:
                    cand_lower = cand.primary_name.lower()
                    raw_lower = raw.text.lower()
                    
                    # 规则 A: 精确匹配或直接子串匹配 (例如 "Turing" 包含于 "Alan Turing")
                    # 添加词边界避免错误包含 (如 "Man" 匹配 "Manchester")
                    if raw_lower == cand_lower or re.search(rf"\b{re.escape(raw_lower)}\b", cand_lower):
                        matched_canonical = cand
                        break
                        
                    # 规则 B: 人名缩写匹配 ("A. Turing" -> "Alan Turing")
                    if label == "Person" and is_initials_match(raw.text, cand.primary_name):
                        matched_canonical = cand
                        break
            
            # 合并或创建新节点
            if matched_canonical:
                mapping[raw.uid] = matched_canonical.uid
                if raw.text != matched_canonical.primary_name:
                    matched_canonical.aliases.add(raw.text)
            else:
                # 成为新的权威节点
                new_uid = f"{label}:{slugify(raw.text)}"
                new_node = CanonicalNode(uid=new_uid, label=label, primary_name=raw.text)
                canonical_candidates.append(new_node)
                nodes_dict[new_uid] = new_node
                mapping[raw.uid] = new_uid

    return list(nodes_dict.values()), mapping

def load_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    p = argparse.ArgumentParser(description="Entity Normalization & Graph Export")
    p.add_argument("--edges-in", default="data/output/relations_rule/edges.csv")
    p.add_argument("--attrs-in", default="data/output/attributes_rule/attrs.csv")
    p.add_argument("--nodes-out", default="data/output/kg_final/nodes.csv")
    p.add_argument("--edges-out", default="data/output/kg_final/edges_normalized.csv")
    p.add_argument("--attrs-out", default="data/output/kg_final/attrs_normalized.csv")
    args = p.parse_args()

    edges_in = Path(args.edges_in)
    attrs_in = Path(args.attrs_in)
    
    Path(args.nodes_out).parent.mkdir(parents=True, exist_ok=True)

    print("1. Loading raw data...")
    raw_edges = load_csv(edges_in)
    raw_attrs = load_csv(attrs_in)

    # 收集所有独特的实体
    entity_counts: Dict[str, RawEntity] = {}
    
    def add_entity(uid: str, label: str, text: str):
        if not uid or not label: return
        if uid not in entity_counts:
            entity_counts[uid] = RawEntity(uid=uid, label=label, text=text, frequency=0)
        entity_counts[uid].frequency += 1

    for row in raw_edges:
        add_entity(row.get("start_uid", ""), row.get("start_label", row["start_uid"].split(":")[0]), row.get("start_uid", "").split(":")[-1].replace("_", " "))
        add_entity(row.get("end_uid", ""), row.get("end_label", row["end_uid"].split(":")[0]), row.get("end_uid", "").split(":")[-1].replace("_", " "))

    for row in raw_attrs:
        add_entity(row.get("entity_uid", ""), row.get("entity_label", row["entity_uid"].split(":")[0]), row.get("entity_text", ""))

    print(f"   Found {len(entity_counts)} unique raw entities.")

    # 聚类消歧
    print("2. Resolving entities...")
    canonical_nodes, uid_mapping = resolve_entities(list(entity_counts.values()))
    
    print(f"   Merged into {len(canonical_nodes)} canonical nodes.")

    # 导出节点表 (Nodes)
    print("3. Exporting Normalized Nodes...")
    with open(args.nodes_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "name", "aliases"])
        for node in canonical_nodes:
            # Join aliases with a pipe '|' for easy Neo4j import
            aliases_str = "|".join(sorted(node.aliases))
            writer.writerow([node.uid, node.label, node.primary_name, aliases_str])

    # 重写边表 (Edges)
    print("4. Exporting Normalized Edges...")
    # 去重处理，避免实体合并后出现完全重复的边
    seen_edges = set()
    with open(args.edges_out, "w", encoding="utf-8", newline="") as f:
        if raw_edges:
            writer = csv.DictWriter(f, fieldnames=list(raw_edges[0].keys()))
            writer.writeheader()
            for row in raw_edges:
                norm_start = uid_mapping.get(row["start_uid"])
                norm_end = uid_mapping.get(row["end_uid"])
                if norm_start and norm_end:
                    row["start_uid"] = norm_start
                    row["end_uid"] = norm_end
                    
                    # 生成边的唯一指纹，如果已经存在则不写入
                    edge_sig = (norm_start, row["relation"], norm_end)
                    if edge_sig not in seen_edges:
                        seen_edges.add(edge_sig)
                        writer.writerow(row)

    # 重写属性表 (Attributes)
    print("5. Exporting Normalized Attributes...")
    seen_attrs = set()
    with open(args.attrs_out, "w", encoding="utf-8", newline="") as f:
        if raw_attrs:
            writer = csv.DictWriter(f, fieldnames=list(raw_attrs[0].keys()))
            writer.writeheader()
            for row in raw_attrs:
                norm_uid = uid_mapping.get(row["entity_uid"])
                if norm_uid:
                    row["entity_uid"] = norm_uid
                    
                    # 生成属性指纹去重 (避免合并实体后出现多条相同的生日记录)
                    attr_sig = (norm_uid, row["attribute_name"], row["attribute_value"])
                    if attr_sig not in seen_attrs:
                        seen_attrs.add(attr_sig)
                        writer.writerow(row)

    print("========================================")
    print(f"Nodes exported: {args.nodes_out}")
    print(f"Edges exported: {args.edges_out}")
    print(f"Attributes exported: {args.attrs_out}")
    print("Graph DB Import Ready!")

if __name__ == "__main__":
    main()