from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Callable

import torch

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# Fallback import for robustness
try:
    import train_ner_bilstm_crf as ner  # type: ignore
except Exception:  # pragma: no cover
    import scripts.train_ner_bilstm_crf as ner  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE_SEED_MATCHERS = ner.build_matchers(
    [(lbl, n) for lbl, names in ner.SCHEMA_SEEDS.items() for n in names],
    case_insensitive=True,
    require_word_boundary=True,
)

_ORG_HINT_RE = re.compile(
    r"\b(University|College|School|Institute|Laboratory|Lab|Committee|Office|Agency|Department|Ministry|Academy|Centre|Center|Library|Park)\b"
)
_ABBR_BOUNDARY_RE = re.compile(r"\b(e\.g|i\.e|mr|mrs|ms|dr|prof)\.$", flags=re.IGNORECASE)
_CAPITALIZED_WORDS_RE = re.compile(r"\b[A-Z][A-Za-z]+\b")
_CLAUSE_BOUNDARY_RE = re.compile(r"\b(while|when|where|which|that|who|because)\b", flags=re.IGNORECASE)
_COORD_BOUNDARY_RE = re.compile(r"\b(and then|and later|and in|and during)\b", flags=re.IGNORECASE)

# 关系触发词正则（高精度，句内触发）
BORN_PATTERNS = [(re.compile(r"\bborn\s+in\b", flags=re.IGNORECASE), "born in")]
DIED_PATTERNS = [(re.compile(r"\bdied\s+in\b", flags=re.IGNORECASE), "died in")]

EDU_PATTERNS = [
    (re.compile(r"\beducated\s+at\b", flags=re.IGNORECASE), "educated at"),
    (re.compile(r"\bgraduated\s+from\b", flags=re.IGNORECASE), "graduated from"),
    (re.compile(r"\bstudied\s+at\b", flags=re.IGNORECASE), "studied at"),
    (re.compile(r"\bstudy\b.{0,60}?\bat\b", flags=re.IGNORECASE), "study ... at"),
    (re.compile(r"\bearned\b.{0,40}?\bdegree\b.{0,20}?\bfrom\b", flags=re.IGNORECASE), "earned degree from"),
]

WORK_PATTERNS = [
    (re.compile(r"\bworked\s+at\b", flags=re.IGNORECASE), "worked at"),
    (re.compile(r"\bworked\s+for\b", flags=re.IGNORECASE), "worked for"),
    (re.compile(r"\bjoined\b.{0,40}?\bat\b", flags=re.IGNORECASE), "joined ... at"),
]

COLLEAGUE_PATTERNS = [
    (re.compile(r"\bcolleague\s+of\b", flags=re.IGNORECASE), "colleague of"),
    (re.compile(r"\bcollaborated\s+with\b", flags=re.IGNORECASE), "collaborated with"),
    (re.compile(r"\bworked\s+with\b", flags=re.IGNORECASE), "worked with"),
]

FRIEND_PATTERNS = [
    (re.compile(r"\bfriend\s+of\b", flags=re.IGNORECASE), "friend of"),
    (re.compile(r"\bfriends\s+with\b", flags=re.IGNORECASE), "friends with"),
    (re.compile(r"\bclose\s+friend\b", flags=re.IGNORECASE), "close friend"),
]

PROPOSED_PATTERNS = [
    (re.compile(r"\bproposed\b", flags=re.IGNORECASE), "proposed"),
    (re.compile(r"\bintroduced\b", flags=re.IGNORECASE), "introduced"),
    (re.compile(r"\bformulated\b", flags=re.IGNORECASE), "formulated"),
]
PROPOSED_PASSIVE_PATTERNS = [
    (re.compile(r"\bwas\s+proposed\s+by\b", flags=re.IGNORECASE), "was proposed by"),
    (re.compile(r"\bwas\s+introduced\s+by\b", flags=re.IGNORECASE), "was introduced by"),
]

INVENTED_PATTERNS = [
    (re.compile(r"\binvented\b", flags=re.IGNORECASE), "invented"),
    (re.compile(r"\bdesigned\b", flags=re.IGNORECASE), "designed"),
    (re.compile(r"\bbuilt\b", flags=re.IGNORECASE), "built"),
]
INVENTED_PASSIVE_PATTERNS = [
    (re.compile(r"\bwas\s+invented\s+by\b", flags=re.IGNORECASE), "was invented by"),
    (re.compile(r"\bwas\s+designed\s+by\b", flags=re.IGNORECASE), "was designed by"),
    (re.compile(r"\bwas\s+built\s+by\b", flags=re.IGNORECASE), "was built by"),
]

AUTHORED_PATTERNS = [
    (re.compile(r"\bauthored\b", flags=re.IGNORECASE), "authored"),
    (re.compile(r"\bwrote\b", flags=re.IGNORECASE), "wrote"),
    (re.compile(r"\bpublished\b.{0,40}?\b(paper|article|report|book)\b", flags=re.IGNORECASE), "published ... paper"),
]
AUTHORED_PASSIVE_PATTERNS = [
    (re.compile(r"\bwas\s+written\s+by\b", flags=re.IGNORECASE), "was written by"),
    (re.compile(r"\bwas\s+authored\s+by\b", flags=re.IGNORECASE), "was authored by"),
]

WORKED_ON_PATTERNS = [
    (re.compile(r"\bworked\s+on\b", flags=re.IGNORECASE), "worked on"),
    (re.compile(r"\bcontributed\s+to\b", flags=re.IGNORECASE), "contributed to"),
    (re.compile(r"\bresearch\b.{0,20}?\bon\b", flags=re.IGNORECASE), "research ... on"),
]

PARTICIPATED_PATTERNS = [
    (re.compile(r"\bparticipated\s+in\b", flags=re.IGNORECASE), "participated in"),
    (re.compile(r"\btook\s+part\s+in\b", flags=re.IGNORECASE), "took part in"),
]

AFFECTED_PATTERNS = [
    (re.compile(r"\bwas\s+affected\s+by\b", flags=re.IGNORECASE), "was affected by"),
    (re.compile(r"\bwas\s+prosecuted\b", flags=re.IGNORECASE), "was prosecuted"),
    (re.compile(r"\bwas\s+convicted\b", flags=re.IGNORECASE), "was convicted"),
    (re.compile(r"\bwas\s+pardoned\b", flags=re.IGNORECASE), "was pardoned"),
]

AWARDED_PATTERNS = [
    (re.compile(r"\bwas\s+awarded\b", flags=re.IGNORECASE), "was awarded"),
    (re.compile(r"\breceived\b.{0,40}?\baward\b", flags=re.IGNORECASE), "received ... award"),
    (re.compile(r"\bwon\b.{0,40}?\baward\b", flags=re.IGNORECASE), "won ... award"),
]

NAMED_PAT = re.compile(r"\bnamed\s+after\b", flags=re.IGNORECASE)

EDGE_CSV_FIELDS = [
    "start_uid", "end_uid", "relation", "evidence", "source", 
    "confidence", "extract_method", "disputed", "start_time", 
    "end_time", "evidence_count"
]


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_ner_model(model_dir: Path, device: torch.device) -> Tuple[ner.BiLSTMCRF, ner.Vocab, ner.Vocab]:
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


def slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.strip().lower()).strip('_')


@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    label: str
    text: str


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int


def build_row_matchers(row: dict) -> List[Tuple[str, str, re.Pattern[str]]]:
    seed_label = str(row.get("seed_label", "")).strip()
    seed_title = str(row.get("seed_title", "")).strip()
    title = str(row.get("title", "")).strip()

    def variants(label: str, surface: str) -> List[str]:
        s = surface.strip()
        if not s:
            return []

        out: List[str] = [s]
        
        # Drop leading article
        if s.lower().startswith("the ") and len(s) > 4:
            out.append(s[4:])

        # Remove parenthetical qualifiers
        no_paren = re.sub(r"\s*\([^)]*\)", "", s).strip()
        if no_paren and no_paren != s:
            out.append(no_paren)

        # Comma qualifier
        if "," in s:
            left = s.split(",", 1)[0].strip()
            if left and left != s:
                out.append(left)

        # Surname for Person
        if label == "Person":
            tokens = [t for t in re.split(r"\s+", no_paren or s) if t]
            if tokens:
                last = tokens[-1].strip(".,;:()[]{}\"'")
                if last.isalpha() and len(last) >= 4:
                    out.append(last)

        # Deduplicate preserving order
        seen = set()
        return [x for x in out if len(x) >= 3 and not (x in seen or seen.add(x))]

    row_entries: List[Tuple[str, str]] = []
    if seed_label and seed_title:
        row_entries.extend((seed_label, v) for v in variants(seed_label, seed_title))
    if seed_label and title and title != seed_title:
        row_entries.extend((seed_label, v) for v in variants(seed_label, title))

    if not row_entries:
        return []

    return ner.build_matchers(
        row_entries,
        case_insensitive=True,
        require_word_boundary=True,
    )


def match_exact_entities(text: str, matchers: Sequence[Tuple[str, str, re.Pattern[str]]]) -> List[Entity]:
    if not text:
        return []

    uniq: Dict[Tuple[str, str], Tuple[str, str, re.Pattern[str]]] = {
        (label, surface): (label, surface, pat) 
        for label, surface, pat in matchers
    }
    ordered = sorted(uniq.values(), key=lambda x: (len(x[1]), x[1].lower()), reverse=True)

    spans = ner.find_non_overlapping_spans(text, ordered)
    
    out = [
        Entity(start=sp.start, end=sp.end, label=sp.label, text=text[sp.start : sp.end])
        for sp in spans if text[sp.start : sp.end].strip()
    ]
    return sorted({(e.start, e.end, e.label, e.text): e for e in out}.values(), key=lambda e: (e.start, e.end, e.label))


def merge_entities(preferred: Sequence[Entity], fallback: Sequence[Entity]) -> List[Entity]:
    """Merge entities using Set intersections (Much faster than arrays)."""
    taken = set()
    merged: List[Entity] = []

    for e in preferred:
        if e.start < e.end:
            taken.update(range(e.start, e.end))
            merged.append(e)

    for e in fallback:
        if e.start < e.end:
            e_range = set(range(e.start, e.end))
            if not taken.intersection(e_range):
                taken.update(e_range)
                merged.append(e)

    return sorted({(e.start, e.end, e.label, e.text): e for e in merged}.values(), key=lambda x: (x.start, x.end, x.label))


def split_into_sentences(text: str) -> List[Sentence]:
    if not text:
        return []

    sentences: List[Sentence] = []
    start = 0
    n = len(text)

    def is_boundary(i: int) -> bool:
        if text[i] not in ".!?":
            return False
        window = text[max(0, i - 6) : i + 1]
        if _ABBR_BOUNDARY_RE.search(window):
            return False
        
        j = i + 1
        while j < n and text[j].isspace():
            j += 1
        if j >= n:
            return True
        nxt = text[j]
        return nxt.isupper() or nxt.isdigit() or nxt in {'"', "'"}

    for i in range(n):
        if is_boundary(i):
            end = i + 1
            sent = text[start:end].strip()
            if sent:
                lstrip = len(text[start:end]) - len(text[start:end].lstrip())
                rstrip = len(text[start:end]) - len(text[start:end].rstrip())
                actual_start = start + lstrip
                actual_end = end - rstrip
                sentences.append(Sentence(text=text[actual_start:actual_end], start=actual_start, end=actual_end))
            start = end

    if start < n:
        tail = text[start:].strip()
        if tail:
            lstrip = len(text[start:]) - len(text[start:].lstrip())
            rstrip = len(text[start:]) - len(text[start:].rstrip())
            sentences.append(Sentence(text=text[start + lstrip : n - rstrip], start=start + lstrip, end=n - rstrip))

    return sentences


@torch.no_grad()
def predict_entities(model: ner.BiLSTMCRF, char_vocab: ner.CharVocab, tag_vocab: ner.TagVocab, text: str, device: torch.device) -> List[Entity]:
    if not text:
        return []

    x = torch.tensor([char_vocab.encode(text)], dtype=torch.long, device=device)
    lengths = torch.tensor([len(text)], dtype=torch.long, device=device)
    mask = torch.ones((1, len(text)), dtype=torch.bool, device=device)

    pred_ids = model.decode(x, mask, lengths)[0]
    pred_tags = tag_vocab.decode(pred_ids)

    spans = ner.bio_to_spans(pred_tags)
    out: List[Entity] = []
    for sp in spans:
        seg = text[sp.start : sp.end]
        if not seg.strip():
            continue
        left_trim = len(seg) - len(seg.lstrip())
        right_trim = len(seg) - len(seg.rstrip())
        start = sp.start + left_trim
        end = sp.end - right_trim
        if start < end:
            out.append(Entity(start=start, end=end, label=sp.label, text=text[start:end]))

    return sorted({(e.start, e.end, e.label, e.text): e for e in out}.values(), key=lambda e: (e.start, e.end, e.label))


def _find_all(pattern: re.Pattern[str], text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def _best_entity(
    entities: Sequence[Entity],
    allowed_labels: Sequence[str],
    *,
    prefer_after: Optional[int] = None,
    prefer_before: Optional[int] = None,
    max_distance: int = 120,
) -> Optional[Entity]:
    allowed_set = set(allowed_labels)
    cands = [e for e in entities if e.label in allowed_set]
    if not cands:
        return None

    def score(e: Entity) -> Tuple[int, int]:
        dist = 10**9
        if prefer_after is not None:
            dist = abs(e.start - prefer_after) + (60 if e.start < prefer_after else 0)
        elif prefer_before is not None:
            dist = abs(prefer_before - e.end) + (60 if e.end > prefer_before else 0)
        return (dist, e.start)

    best = min(cands, key=score)
    if score(best)[0] > max_distance:
        return None
    return best


def _clip_object_phrase(tail: str) -> str:
    s = tail.strip()
    if not s: return ""
    s = re.sub(r"^[\s\-:;,.\"'()\[\]]+", "", s)
    s = re.sub(r"^(the|a|an)\s+", "", s, flags=re.IGNORECASE)
    
    m = re.search(r"[.!?]", s)
    if m: s = s[: m.start()]

    m = _CLAUSE_BOUNDARY_RE.search(s)
    if m: s = s[: m.start()]

    m = _COORD_BOUNDARY_RE.search(s)
    if m: s = s[: m.start()]

    return s.strip(" \t\r\n\"'()[]{};:")


def _make_object_entity(sentence: str, trigger_end: int, label: str, phrase: str) -> Optional[Entity]:
    phrase = phrase.strip()
    if not phrase: return None
    start = sentence.find(phrase, trigger_end)
    start = start if start >= 0 else trigger_end
    return Entity(start=start, end=start + len(phrase), label=label, text=phrase)


def _heuristic_org_after(sentence: str, trigger_end: int) -> Optional[Entity]:
    phrase = _clip_object_phrase(sentence[trigger_end:])[:120].strip()
    if not phrase: return None
    if not _ORG_HINT_RE.search(phrase) and len(_CAPITALIZED_WORDS_RE.findall(phrase)) < 2:
        return None
    return _make_object_entity(sentence, trigger_end, "Organization", phrase)


def _heuristic_loc_after(sentence: str, trigger_end: int) -> Optional[Entity]:
    phrase = _clip_object_phrase(sentence[trigger_end:])[:120].strip()
    if not phrase or not _CAPITALIZED_WORDS_RE.search(phrase): return None
    return _make_object_entity(sentence, trigger_end, "Location", phrase)


# ==========================================
# 优化点 3: 提取关系判断通用逻辑，消除冗余代码
# ==========================================
def _extract_std_relation(
    sentence: str, entities: Sequence[Entity], subject_person: Optional[Entity], relaxed: bool,
    patterns: List[Tuple[re.Pattern, str]], target_label: str, rel_name: str, heuristic_fn: Callable
) -> List[dict]:
    edges = []
    for pat, trig in patterns:
        for t_start, t_end in _find_all(pat, sentence):
            person = _best_entity(entities, ["Person"], prefer_before=t_start) or \
                     _best_entity(entities, ["Person"], prefer_after=t_end) or \
                     subject_person
            target = _best_entity(entities, [target_label], prefer_after=t_end)
            if target is None and relaxed:
                target = heuristic_fn(sentence, t_end)
            
            if person and target:
                edges.append({
                    "relation": rel_name, "start": person, "end": target, 
                    "trigger": trig, "confidence": "high"
                })
    return edges


def _extract_person_target_from_passive(
    sentence: str,
    entities: Sequence[Entity],
    patterns: List[Tuple[re.Pattern, str]],
    target_label: str,
    rel_name: str,
) -> List[dict]:
    """Handle passive constructions like "X was written by Y" -> Y (Person) -> X (target)."""

    edges: List[dict] = []
    for pat, trig in patterns:
        for t_start, t_end in _find_all(pat, sentence):
            target = _best_entity(entities, [target_label], prefer_before=t_start)
            person = _best_entity(entities, ["Person"], prefer_after=t_end)
            if person and target:
                edges.append({
                    "relation": rel_name,
                    "start": person,
                    "end": target,
                    "trigger": trig,
                    "confidence": "high",
                })
    return edges


def _extract_person_person_relation(
    sentence: str,
    entities: Sequence[Entity],
    patterns: List[Tuple[re.Pattern, str]],
    rel_name: str,
) -> List[dict]:
    edges: List[dict] = []
    for pat, trig in patterns:
        for t_start, t_end in _find_all(pat, sentence):
            left = _best_entity(entities, ["Person"], prefer_before=t_start)
            right = _best_entity(entities, ["Person"], prefer_after=t_end)
            if not left or not right:
                continue
            if left.start == right.start and left.end == right.end and left.text == right.text:
                continue
            edges.append({
                "relation": rel_name,
                "start": left,
                "end": right,
                "trigger": trig,
                "confidence": "high",
            })
    return edges


def _extract_subject_event_relation(
    sentence: str,
    entities: Sequence[Entity],
    subject_labels: Sequence[str],
    patterns: List[Tuple[re.Pattern, str]],
    rel_name: str,
    *,
    fallback_person: Optional[Entity] = None,
) -> List[dict]:
    edges: List[dict] = []
    for pat, trig in patterns:
        for t_start, t_end in _find_all(pat, sentence):
            subject = (
                _best_entity(entities, list(subject_labels), prefer_before=t_start)
                or _best_entity(entities, list(subject_labels), prefer_after=t_end)
            )
            if subject is None and fallback_person is not None and "Person" in subject_labels:
                subject = fallback_person

            event = _best_entity(entities, ["Event"], prefer_after=t_end) or _best_entity(entities, ["Event"], prefer_before=t_start)
            if subject and event:
                edges.append({
                    "relation": rel_name,
                    "start": subject,
                    "end": event,
                    "trigger": trig,
                    "confidence": "high",
                })
    return edges


def extract_relations_from_sentence(
    sentence: str, entities: Sequence[Entity], *, subject_person: Optional[Entity] = None, relaxed: bool = False
) -> List[dict]:
    if not sentence or not entities:
        return []

    edges: List[dict] = []
    no_heuristic = lambda _s, _i: None

    # 3.1 Social & Life
    edges.extend(_extract_std_relation(sentence, entities, subject_person, relaxed, BORN_PATTERNS, "Location", "BORN_IN", _heuristic_loc_after))
    edges.extend(_extract_std_relation(sentence, entities, subject_person, relaxed, DIED_PATTERNS, "Location", "DIED_IN", _heuristic_loc_after))
    edges.extend(_extract_std_relation(sentence, entities, subject_person, relaxed, EDU_PATTERNS, "Organization", "EDUCATED_AT", _heuristic_org_after))
    edges.extend(_extract_std_relation(sentence, entities, subject_person, relaxed, WORK_PATTERNS, "Organization", "WORKED_AT", _heuristic_org_after))
    edges.extend(_extract_person_person_relation(sentence, entities, COLLEAGUE_PATTERNS, "COLLEAGUE_OF"))
    edges.extend(_extract_person_person_relation(sentence, entities, FRIEND_PATTERNS, "FRIEND_OF"))

    # 3.2 Academic & Achievements
    edges.extend(_extract_std_relation(sentence, entities, subject_person, relaxed, PROPOSED_PATTERNS, "Concept", "PROPOSED", no_heuristic))
    edges.extend(_extract_person_target_from_passive(sentence, entities, PROPOSED_PASSIVE_PATTERNS, "Concept", "PROPOSED"))

    edges.extend(_extract_std_relation(sentence, entities, subject_person, relaxed, INVENTED_PATTERNS, "Artifact", "INVENTED", no_heuristic))
    edges.extend(_extract_person_target_from_passive(sentence, entities, INVENTED_PASSIVE_PATTERNS, "Artifact", "INVENTED"))

    edges.extend(_extract_std_relation(sentence, entities, subject_person, relaxed, AUTHORED_PATTERNS, "Publication", "AUTHORED", no_heuristic))
    edges.extend(_extract_person_target_from_passive(sentence, entities, AUTHORED_PASSIVE_PATTERNS, "Publication", "AUTHORED"))

    edges.extend(_extract_std_relation(sentence, entities, subject_person, relaxed, WORKED_ON_PATTERNS, "Artifact", "WORKED_ON", no_heuristic))
    edges.extend(_extract_std_relation(sentence, entities, subject_person, relaxed, WORKED_ON_PATTERNS, "Concept", "WORKED_ON", no_heuristic))

    # 3.3 Events & Honors
    edges.extend(_extract_subject_event_relation(sentence, entities, ["Person", "Organization"], PARTICIPATED_PATTERNS, "PARTICIPATED_IN", fallback_person=subject_person))
    edges.extend(_extract_subject_event_relation(sentence, entities, ["Person"], AFFECTED_PATTERNS, "AFFECTED_BY", fallback_person=subject_person))
    edges.extend(_extract_std_relation(sentence, entities, subject_person, relaxed, AWARDED_PATTERNS, "Honor", "AWARDED", no_heuristic))

    # NAMED_AFTER: (Honor|Organization) -> Person
    for t_start, t_end in _find_all(NAMED_PAT, sentence):
        left = _best_entity(entities, ["Honor", "Organization"], prefer_before=t_start)
        right = _best_entity(entities, ["Person"], prefer_after=t_end)
        if left and right:
            edges.append({
                "relation": "NAMED_AFTER",
                "start": left,
                "end": right,
                "trigger": "named after",
                "confidence": "high",
            })

    # Deduplicate by (relation, start_uid, end_uid)
    dedup = {
        (e["relation"], f"{e['start'].label}:{slugify(e['start'].text)}", f"{e['end'].label}:{slugify(e['end'].text)}"): e
        for e in edges
    }

    return [
        {
            "relation": e["relation"],
            "start_label": e["start"].label,
            "start_text": e["start"].text,
            "start_uid": f"{e['start'].label}:{slugify(e['start'].text)}",
            "end_label": e["end"].label,
            "end_text": e["end"].text,
            "end_uid": f"{e['end'].label}:{slugify(e['end'].text)}",
            "trigger": e["trigger"],
            "confidence": e["confidence"],
        }
        for e in dedup.values()
    ]


def extract_from_record(
    row: dict, *, model: ner.BiLSTMCRF, char_vocab: ner.CharVocab, tag_vocab: ner.TagVocab, 
    device: torch.device, relaxed: bool
) -> List[dict]:
    text = str(row.get("text", ""))
    if not text: return []

    source = str(row.get("url", "")) or str(row.get("uid", "")) or str(row.get("source", ""))
    doc_id = str(row.get("doc_id", row.get("id", "")))

    row_matchers = build_row_matchers(row)
    exact_matchers = list(row_matchers) + list(BASE_SEED_MATCHERS)

    out: List[dict] = []
    for sent_idx, sent in enumerate(split_into_sentences(text)):
        exact_ents = match_exact_entities(sent.text, exact_matchers)
        pred_ents = predict_entities(model, char_vocab, tag_vocab, sent.text, device)
        ents = merge_entities(exact_ents, pred_ents)
        if not ents: continue

        subject_person = None
        if row.get("seed_label") == "Person":
            seed_title = str(row.get("seed_title", "")).strip()
            if seed_title and not any(e.label == "Person" for e in ents):
                if re.match(r"^(He|She|His|Her)\b", sent.text.strip()):
                    subject_person = Entity(start=-1, end=-1, label="Person", text=seed_title)

        for e in extract_relations_from_sentence(sent.text, ents, subject_person=subject_person, relaxed=relaxed):
            out.append({
                "doc_id": doc_id,
                "sent_id": f"{doc_id}__sent{sent_idx:03d}" if doc_id else f"sent{sent_idx:03d}",
                "relation": e["relation"], "start_uid": e["start_uid"], "end_uid": e["end_uid"],
                "start_label": e["start_label"], "end_label": e["end_label"],
                "start_text": e["start_text"], "end_text": e["end_text"],
                "evidence": sent.text, "source": source, "confidence": e["confidence"],
                "extract_method": "rule", "trigger": e["trigger"],
            })
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rule-based relation extraction (schema-constrained)")
    p.add_argument("--input", default="data/raw/turing_schema_corpus.jsonl", help="Input corpus JSONL")
    p.add_argument("--model-dir", default="data/output/ner_bilstm_crf", help="NER model directory")
    p.add_argument("--output-csv", default="data/output/relations_rule/edges.csv", help="Output edges CSV")
    p.add_argument("--output-jsonl", default="data/output/relations_rule/edges.jsonl", help="Output edges JSONL")

    # 默认启用 CUDA
    p.add_argument("--cuda", dest="cuda", action="store_true", help="Enable CUDA if available (default: enabled)")
    p.add_argument("--no-cuda", dest="cuda", action="store_false", help="Disable CUDA and force CPU")
    p.set_defaults(cuda=True)

    p.add_argument("--limit", type=int, default=0, help="Optional max records to process (0=all)")
    p.add_argument("--relaxed", action="store_true", help="Relax object constraints (higher recall, lower precision)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = (REPO_ROOT / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    model_dir = (REPO_ROOT / args.model_dir).resolve() if not Path(args.model_dir).is_absolute() else Path(args.model_dir)
    output_csv = (REPO_ROOT / args.output_csv).resolve() if not Path(args.output_csv).is_absolute() else Path(args.output_csv)
    output_jsonl = (REPO_ROOT / args.output_jsonl).resolve() if not Path(args.output_jsonl).is_absolute() else Path(args.output_jsonl)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model, char_vocab, tag_vocab = load_ner_model(model_dir, device)

    processed = 0
    total_edges = 0

    with output_jsonl.open("w", encoding="utf-8") as f_jsonl, \
         output_csv.open("w", encoding="utf-8", newline="") as f_csv:
        
        csv_writer = csv.DictWriter(f_csv, fieldnames=EDGE_CSV_FIELDS)
        csv_writer.writeheader()

        total_records = None
        if tqdm is not None:
            if args.limit > 0:
                total_records = int(args.limit)
            else:
                try:
                    with input_path.open("r", encoding="utf-8") as f_tmp:
                        total_records = sum(1 for line in f_tmp if line.strip())
                except Exception:
                    total_records = None

        rows = iter_jsonl(input_path)

        pbar = None
        if tqdm is not None:
            pbar = tqdm(rows, total=total_records, desc="Extracting relations", unit="doc")
            rows = pbar

        try:
            for row in rows:
                processed += 1
                if 0 < args.limit < processed:
                    processed -= 1  # 恢复实际处理数目
                    break

                edges = extract_from_record(
                    row,
                    model=model,
                    char_vocab=char_vocab,
                    tag_vocab=tag_vocab,
                    device=device,
                    relaxed=args.relaxed,
                )

                for e in edges:
                    # 写入 JSONL
                    f_jsonl.write(json.dumps(e, ensure_ascii=False) + "\n")
                    # 写入 CSV
                    csv_writer.writerow({
                        "start_uid": e["start_uid"],
                        "end_uid": e["end_uid"],
                        "relation": e["relation"],
                        "evidence": e["evidence"],
                        "source": e["source"],
                        "confidence": e["confidence"],
                        "extract_method": e["extract_method"],
                        "disputed": "false",
                        "start_time": "",
                        "end_time": "",
                        "evidence_count": "1",
                    })
                total_edges += len(edges)

                if pbar is not None:
                    pbar.set_postfix(edges=total_edges)
        finally:
            if pbar is not None:
                pbar.close()

    print(f"[DONE] processed_records={processed} edges={total_edges}")
    print(f"[DONE] edges_jsonl={output_jsonl}")
    print(f"[DONE] edges_csv={output_csv}")


if __name__ == "__main__":
    main()