from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# 复用已有的 NER 模块
try:
    import train_ner_bilstm_crf as ner  # type: ignore
except Exception:  # pragma: no cover
    import scripts.train_ner_bilstm_crf as ner  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_ner_model(model_dir: Path, device: torch.device):
    """Load the BiLSTM-CRF NER model produced by scripts/train_ner_bilstm_crf.py."""

    with (model_dir / "char_vocab.json").open("r", encoding="utf-8") as f1:
        char_vocab = ner.Vocab.from_dict(json.load(f1))
    with (model_dir / "tag_vocab.json").open("r", encoding="utf-8") as f2:
        tag_vocab = ner.Vocab.from_dict(json.load(f2))

    try:
        ckpt = torch.load(model_dir / "model.pt", map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(model_dir / "model.pt", map_location=device)

    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = ner.BiLSTMCRF(
        len(char_vocab.itos),
        len(tag_vocab.itos),
        int(cfg.get("embedding_dim", 64)),
        int(cfg.get("hidden_dim", 128)),
        int(cfg.get("num_layers", 1)),
        float(cfg.get("dropout", 0.1)),
        char_vocab.pad_id,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, char_vocab, tag_vocab


# ==========================================
# 1. 全局正则表达式与词典 (提升性能)
# ==========================================

# 匹配日期 (如: 23 June 1912, June 23, 1912, 1912)
_DATE_REGEX_STR = r"\b(?:(?:[0-3]?\d\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+)?\d{4}\b"
_DATE_RE = re.compile(_DATE_REGEX_STR, flags=re.IGNORECASE)

# 匹配生卒年份括号，如 Alan Turing (23 June 1912 – 7 June 1954)
_LIFE_SPAN_RE = re.compile(r"\(\s*([^\-–]+)?(?:born\s+)?(" + _DATE_REGEX_STR + r")\s*[-–]\s*(?:died\s+)?(" + _DATE_REGEX_STR + r")\s*\)", flags=re.IGNORECASE)

# 别名/曾用名触发词
_ALIAS_RE = re.compile(r"\b(?:also known as|a\.k\.a\.|alias|born as|pen name)\s+([A-Z][a-zA-Z\s]+(?:'[A-Z][a-zA-Z\s]+')?)", flags=re.IGNORECASE)

# 机构成立触发词
_FOUNDED_RE = re.compile(r"\b(?:founded|established|formed|created)\s+(?:in|on)\s+(" + _DATE_REGEX_STR + r")", flags=re.IGNORECASE)

# 图灵领域常见职业词典
TURING_OCCUPATIONS = {
    "mathematician",
    "computer scientist",
    "logician",
    "cryptanalyst",
    "philosopher",
    "theoretical biologist",
    "physicist",
    "engineer",
    "programmer",
    "codebreaker",
    "professor",
    "researcher",
    "author",
}
_OCCUPATION_RE = re.compile(
    r"\b(" + "|".join(re.escape(x) for x in sorted(TURING_OCCUPATIONS, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

# 常见国籍（最小覆盖，用于 Person.nationality）
TURING_NATIONALITIES = {
    "British",
    "English",
    "American",
    "Hungarian",
    "German",
    "Polish",
    "Dutch",
    "French",
    "Italian",
    "Russian",
    "Austrian",
    "Canadian",
}
_NATIONALITY_RE = re.compile(
    r"\b(" + "|".join(re.escape(x) for x in sorted(TURING_NATIONALITIES, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

# 常见组织类型关键词（最小覆盖，用于 Organization.org_type）
_ORG_TYPE_MAP = {
    "university": "university",
    "college": "college",
    "school": "school",
    "institute": "institute",
    "laboratory": "laboratory",
    "lab": "laboratory",
    "department": "department",
    "ministry": "ministry",
    "office": "office",
    "agency": "agency",
    "committee": "committee",
    "academy": "academy",
    "centre": "centre",
    "center": "centre",
    "park": "park",
    "government": "government",
}


@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    label: str
    text: str

# 借用之前优化过的工具函数
def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.strip().lower()).strip('_')

def _best_entity(entities: Sequence[Entity], allowed_labels: Sequence[str], prefer_before: int) -> Optional[Entity]:
    allowed_set = set(allowed_labels)
    cands = [e for e in entities if e.label in allowed_set and e.end <= prefer_before]
    if not cands:
        return None
    # 找距离触发词最近的实体
    return min(cands, key=lambda e: prefer_before - e.end)


def _infer_org_type(org_name: str) -> Optional[str]:
    s = org_name.lower()
    for kw, typ in _ORG_TYPE_MAP.items():
        if kw in s:
            return typ
    return None


# ==========================================
# 2. 核心属性抽取逻辑
# ==========================================

def extract_attributes_from_sentence(
    sentence: str, 
    entities: Sequence[Entity], 
    subject_person: Optional[Entity] = None
) -> List[dict]:
    """从单个句子中抽取属性"""
    if not sentence:
        return []

    attributes: List[dict] = []

    # A. 抽取生命周期 (Birth Date, Death Date) - 基于括号模式
    for match in _LIFE_SPAN_RE.finditer(sentence):
        birth_date, death_date = match.group(2), match.group(3)
        # 括号通常紧跟在人名后面
        person = _best_entity(entities, ["Person"], prefer_before=match.start()) or subject_person
        if person:
            if birth_date:
                attributes.append({"entity": person, "attr_name": "birth_date", "attr_value": birth_date.strip()})
            if death_date:
                attributes.append({"entity": person, "attr_name": "death_date", "attr_value": death_date.strip()})

    # B. 抽取显式出生/死亡日期
    for trigger, attr_type in [("born", "birth_date"), ("died", "death_date")]:
        for match in re.finditer(rf"\b{trigger}\b\s+(?:on|in)?\s*({_DATE_REGEX_STR})", sentence, flags=re.IGNORECASE):
            person = _best_entity(entities, ["Person"], prefer_before=match.start()) or subject_person
            if person:
                attributes.append({"entity": person, "attr_name": attr_type, "attr_value": match.group(1).strip()})

    # C. 抽取职业 (Occupation)
    for match in _OCCUPATION_RE.finditer(sentence):
        # 职业通常在人名之后，如 "Alan Turing was a mathematician"
        person = _best_entity(entities, ["Person"], prefer_before=match.start()) or subject_person
        if person:
            attributes.append({"entity": person, "attr_name": "occupation", "attr_value": match.group(1).lower().strip()})

    # C2. 抽取国籍 (Nationality) —— 仅在后文出现职业等线索时才采纳，降低误报
    for match in _NATIONALITY_RE.finditer(sentence):
        tail = sentence[match.end() : match.end() + 80]
        if not (_OCCUPATION_RE.search(tail) or re.search(r"\b(citizen|nationality)\b", tail, flags=re.IGNORECASE)):
            continue
        person = _best_entity(entities, ["Person"], prefer_before=match.start()) or subject_person
        if person:
            attributes.append({"entity": person, "attr_name": "nationality", "attr_value": match.group(1).strip().title()})

    # D. 抽取别名 (Alias)
    for match in _ALIAS_RE.finditer(sentence):
        target_entity = _best_entity(entities, ["Person", "Organization"], prefer_before=match.start()) or subject_person
        if target_entity:
            alias_val = match.group(1).strip()
            # 简单清理末尾可能带入的标点
            alias_val = re.sub(r"[,.;]+$", "", alias_val)
            attributes.append({"entity": target_entity, "attr_name": "aliases", "attr_value": alias_val})

    # E. 抽取机构成立时间 (Foundation Year)
    for match in _FOUNDED_RE.finditer(sentence):
        org = _best_entity(entities, ["Organization"], prefer_before=match.start())
        if org:
            date_str = match.group(1).strip()
            year_m = re.search(r"\b(\d{4})\b", date_str)
            if year_m:
                attributes.append({"entity": org, "attr_name": "established_year", "attr_value": int(year_m.group(1))})

    # F. 抽取组织类型 (org_type) —— 由组织名称推断
    for org in (e for e in entities if e.label == "Organization"):
        typ = _infer_org_type(org.text)
        if typ:
            attributes.append({"entity": org, "attr_name": "org_type", "attr_value": typ})

    # 去重处理
    dedup = {}
    for attr in attributes:
        ent = attr["entity"]
        uid = f"{ent.label}:{slugify(ent.text)}"
        key = (uid, attr["attr_name"], attr["attr_value"])
        if key not in dedup:
            dedup[key] = {
                "entity_uid": uid,
                "entity_label": ent.label,
                "entity_text": ent.text,
                "attribute_name": attr["attr_name"],
                "attribute_value": attr["attr_value"],
                "confidence": "high"
            }

    return list(dedup.values())

# ==========================================
# 3. 记录处理与 IO 编排
# ==========================================

# 简化的分句函数 (复用之前的正则逻辑可进一步提升)
def simple_sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def extract_attrs_from_record(row: dict, model, char_vocab, tag_vocab, device: torch.device) -> List[dict]:
    text = str(row.get("text", ""))
    if not text:
        return []

    doc_id = str(row.get("doc_id", row.get("id", "")))
    source = str(row.get("url", "")) or doc_id

    out: List[dict] = []
    
    for sent_idx, sent_text in enumerate(simple_sentence_split(text)):
        # 1. 运行 NER 提取实体 (使用你在关系抽取中相同的方法)
        # 注意: 这里假设你之前写的 predict_entities 依然可用
        x = torch.tensor([char_vocab.encode(sent_text)], dtype=torch.long, device=device)
        lengths = torch.tensor([len(sent_text)], dtype=torch.long, device=device)
        mask = torch.ones((1, len(sent_text)), dtype=torch.bool, device=device)
        
        pred_ids = model.decode(x, mask, lengths)[0]
        pred_tags = tag_vocab.decode(pred_ids)
        spans = ner.bio_to_spans(pred_tags)
        
        ents = [Entity(start=sp.start, end=sp.end, label=sp.label, text=sent_text[sp.start:sp.end]) for sp in spans]

        # 主语 Fallback 逻辑
        subject_person = None
        if row.get("seed_label") == "Person":
            seed_title = str(row.get("seed_title", "")).strip()
            if seed_title and not any(e.label == "Person" for e in ents):
                subject_person = Entity(start=-1, end=-1, label="Person", text=seed_title)

        # 2. 抽取属性
        attrs = extract_attributes_from_sentence(sent_text, ents, subject_person)
        
        for a in attrs:
            a["doc_id"] = doc_id
            a["sent_id"] = f"{doc_id}__sent{sent_idx:03d}" if doc_id else f"sent{sent_idx:03d}"
            a["evidence"] = sent_text
            a["source"] = source
            a["extract_method"] = "regex_rule"
            out.append(a)

    return out

ATTR_CSV_FIELDS = [
    "entity_uid", "entity_label", "entity_text", 
    "attribute_name", "attribute_value", 
    "evidence", "source", "confidence", "extract_method"
]

def main() -> None:
    p = argparse.ArgumentParser(description="Attribute Extraction for Turing KG")
    p.add_argument("--input", default="data/raw/turing_schema_corpus.jsonl")
    p.add_argument("--model-dir", default="data/output/ner_bilstm_crf")
    p.add_argument("--output-csv", default="data/output/attributes_rule/attrs.csv")
    p.add_argument("--output-jsonl", default="data/output/attributes_rule/attrs.jsonl")

    cuda_group = p.add_mutually_exclusive_group()
    cuda_group.add_argument("--cuda", dest="cuda", action="store_true", help="use CUDA if available")
    cuda_group.add_argument("--no-cuda", dest="cuda", action="store_false", help="force CPU")
    p.set_defaults(cuda=True)

    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    input_path = Path(args.input)
    output_csv = Path(args.output_csv)
    output_jsonl = Path(args.output_jsonl)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # 加载 NER 模型 (用于定位属性归属于哪个实体)
    model, char_vocab, tag_vocab = load_ner_model(Path(args.model_dir), device)

    processed = 0
    total_attrs = 0

    total_records = None
    if tqdm:
        if args.limit > 0:
            total_records = int(args.limit)
        else:
            try:
                with input_path.open("r", encoding="utf-8") as f_tmp:
                    total_records = sum(1 for line in f_tmp if line.strip())
            except Exception:
                total_records = None

    with output_jsonl.open("w", encoding="utf-8") as f_jsonl, \
        output_csv.open("w", encoding="utf-8", newline="") as f_csv:

        csv_writer = csv.DictWriter(f_csv, fieldnames=ATTR_CSV_FIELDS)
        csv_writer.writeheader()

        with input_path.open("r", encoding="utf-8") as f_in:
            pbar = tqdm(total=total_records, desc="Extracting attributes") if tqdm else None
            try:
                for line in f_in:
                    if not line.strip():
                        continue
                    if 0 < args.limit <= processed:
                        break

                    row = json.loads(line)
                    processed += 1

                    attrs = extract_attrs_from_record(row, model, char_vocab, tag_vocab, device)

                    for a in attrs:
                        # JSONL 写入
                        f_jsonl.write(json.dumps(a, ensure_ascii=False) + "\n")
                        # CSV 写入
                        csv_writer.writerow({
                            "entity_uid": a["entity_uid"],
                            "entity_label": a["entity_label"],
                            "entity_text": a["entity_text"],
                            "attribute_name": a["attribute_name"],
                            "attribute_value": a["attribute_value"],
                            "evidence": a["evidence"],
                            "source": a["source"],
                            "confidence": a["confidence"],
                            "extract_method": a["extract_method"],
                        })
                    total_attrs += len(attrs)

                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix(attrs=total_attrs)
            finally:
                if pbar:
                    pbar.close()

    print(f"[DONE] Processed records: {processed}")
    print(f"[DONE] Attributes extracted: {total_attrs}")
    print(f"[DONE] Saved to: {output_csv}")

if __name__ == "__main__":
    main()