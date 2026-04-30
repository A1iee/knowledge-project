import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from tqdm import tqdm

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModelForSequenceClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    TRANSFORMERS_AVAILABLE = False

try:
    import train_ner_bilstm_crf as ner_module
except ImportError:
    import scripts.train_ner_bilstm_crf as ner_module


CENTER_ENTITY_ID = "alan_turing"
CENTER_ENTITY_CANONICAL = "Alan Turing"
CENTER_ENTITY_NAMES = {"alan turing", "alan mathison turing", "turing"}
PAGE_ENTITY_PRONOUNS = {
    "alan turing": {"he", "his"},
    "joan clarke": {"she", "her"},
}

RELATION_SCHEMA = {
    "BORN_IN": {
        "sub": ["Person"],
        "obj": ["Location"],
        "triggers": [r"\bwas born in\b", r"\bborn in\b"],
        "strategy": "subject_before_object_after",
        "mode": "rule_first",
        "rule_confidence": 0.97,
    },
    "DIED_IN": {
        "sub": ["Person"],
        "obj": ["Location"],
        "triggers": [r"\bdied in\b", r"\bpassed away in\b"],
        "strategy": "subject_before_object_after",
        "mode": "rule_first",
        "rule_confidence": 0.94,
    },
    "EDUCATED_AT": {
        "sub": ["Person"],
        "obj": ["Organization"],
        "triggers": [r"\bstudied at\b", r"\bgraduated from\b", r"\bwas educated at\b", r"\battended\b"],
        "strategy": "subject_before_object_after",
        "mode": "rule_first",
        "rule_confidence": 0.96,
    },
    "WORKED_AT": {
        "sub": ["Person"],
        "obj": ["Organization"],
        "triggers": [r"\bworked at\b", r"\bworked for\b", r"\bserved at\b", r"\bjoined\b"],
        "strategy": "subject_before_object_after",
        "mode": "rule_first",
        "rule_confidence": 0.95,
    },
    "COLLEAGUE_OF": {
        "sub": ["Person"],
        "obj": ["Person"],
        "triggers": [r"\bworked with\b", r"\bcollaborated with\b", r"\bcolleague of\b", r"\btogether with\b"],
        "strategy": "person_pair",
        "mode": "rule_first",
        "rule_confidence": 0.92,
    },
    "PROPOSED": {
        "sub": ["Person"],
        "obj": ["Concept"],
        "triggers": [r"\bproposed\b", r"\bformulated\b", r"\bintroduced\b", r"\bdevised\b"],
        "strategy": "subject_before_object_after",
        "mode": "rule_first",
        "rule_confidence": 0.94,
    },
    "WORKED_ON": {
        "sub": ["Person"],
        "obj": ["Concept", "Artifact"],
        "triggers": [r"\bworked on\b", r"\bdesigned\b", r"\bdeveloped\b", r"\bimproved\b", r"\bbroke\b", r"\bbreaking\b"],
        "strategy": "subject_before_object_after",
        "mode": "rule_first",
        "rule_confidence": 0.86,
    },
    "AUTHORED": {
        "sub": ["Person"],
        "obj": ["Publication"],
        "triggers": [r"\bwrote\b", r"\bauthored\b", r"\bpublished\b"],
        "strategy": "subject_before_object_after",
        "mode": "rule_first",
        "rule_confidence": 0.86,
    },
    "PARTICIPATED_IN": {
        "sub": ["Person", "Organization"],
        "obj": ["Event"],
        "triggers": [r"\bparticipated in\b", r"\btook part in\b", r"\bserved in\b", r"\bduring\b"],
        "strategy": "subject_before_object_after",
        "mode": "nli_assist",
        "rule_confidence": 0.72,
    },
    "AFFECTED_BY": {
        "sub": ["Person"],
        "obj": ["Event", "Honor", "Organization"],
        "triggers": [r"\baffected by\b", r"\bprosecuted\b", r"\bconvicted\b", r"\bpardoned\b", r"\bimpacted by\b"],
        "strategy": "subject_before_object_after_or_event_before",
        "mode": "nli_assist",
        "rule_confidence": 0.73,
    },
    "HONORED_BY": {
        "sub": ["Person"],
        "obj": ["Honor", "Organization"],
        "triggers": [r"\bhonored by\b", r"\bcommemorated by\b", r"\brecognized by\b", r"\baward\b", r"\bnote\b"],
        "strategy": "subject_before_object_after_or_object_before",
        "mode": "nli_assist",
        "rule_confidence": 0.72,
    },
    "NAMED_AFTER": {
        "sub": ["Honor", "Organization", "Artifact"],
        "obj": ["Person"],
        "triggers": [r"\bnamed after\b", r"\bin honor of\b", r"\bin recognition of\b"],
        "strategy": "subject_before_object_after",
        "mode": "nli_assist",
        "rule_confidence": 0.75,
    },
}

RULE_FIRST_RELATIONS = {"BORN_IN", "DIED_IN","EDUCATED_AT", "WORKED_AT","COLLEAGUE_OF","PROPOSED"}
NLI_ASSIST_THRESHOLDS = {
    "WORKED_ON": 0.24,
    "AUTHORED": 0.24,
    "PARTICIPATED_IN": 0.55,
    "AFFECTED_BY": 0.45,
    "HONORED_BY": 0.42,
    "NAMED_AFTER": 0.45,
}
HONOR_LABELS = {"Honor", "Award", "Decoration"}
AFFECTED_OBJECT_LABELS = {"Event", "Honor", "Organization"}
HONORED_OBJECT_LABELS = {"Honor", "Award", "Decoration", "Organization"}
NAMED_AFTER_SUBJECT_LABELS = {"Honor", "Organization", "Artifact"}
EDUCATION_PHRASE_HINTS = [
    "earned a doctorate degree from",
    "earned a doctorate from",
    "received a doctorate from",
    "received his doctorate from",
    "received her doctorate from",
    "graduated from",
    "studied at",
    "attended",
]
EDUCATION_ORG_HINTS = {
    "king's college, cambridge",
    "princeton university",
}
WORKED_ON_RULE_TRIGGERS = {
    "designed",
    "developed",
    "improved",
    "worked on",
    "devised",
    "improvements to",
    "contributed to the development of",
}
AUTHORED_RULE_TRIGGERS = {
    "wrote",
    "authored",
    "wrote on",
    "published a paper on",
}
COLLEAGUE_RULE_TRIGGERS = {
    "worked with",
    "collaborated with",
    "lifelong friends and associates",
}
PARTICIPATED_RULE_TRIGGERS = {
    "participated in",
    "took part in",
    "served in",
    "during world war ii",
    "during the second world war",
    "codebreaking during the war",
    "during world war 2",
}
HONORED_RULE_TRIGGERS = {
    "honored by",
    "commemorated by",
    "recognized by",
    "appointed as",
}
NAMED_AFTER_RULE_TRIGGERS = {
    "named after",
    "in honor of",
    "in recognition of",
}
WORKED_ON_TRIGGER_SPECS = [
    ("devised", 0.9),
    ("developed", 0.9),
    ("improvements to", 0.88),
    ("contributed to the development of", 0.9),
    ("responsible for", 0.84),
    ("led the development of", 0.9),
    ("helped develop", 0.88),
]
AUTHORED_TRIGGER_SPECS = [
    ("wrote on", 0.9),
    ("wrote", 0.9),
    ("paper on", 0.84),
    ("published a paper on", 0.9),
    ("in his paper", 0.82),
    ("in her paper", 0.82),
    ("in his 1936 paper", 0.82),
    ("in her 1936 paper", 0.82),
]
COLLEAGUE_TRIGGER_SPECS = [
    ("lifelong friends and associates", 0.88),
    ("friends and associates", 0.84),
    ("worked with", 0.88),
    ("collaborated with", 0.88),
    ("alongside", 0.82),
    ("together with", 0.84),
    ("one of turing's lifelong friends and associates", 0.88),
]
PARTICIPATED_TRIGGER_SPECS = [
    ("during world war ii", 0.9),
    ("during the second world war", 0.9),
    ("during the war", 0.84),
    ("wartime", 0.84),
    ("codebreaking during the war", 0.88),
    ("during world war 2", 0.9),
]
HONOR_TRIGGER_SPECS = [
    ("awarded", 0.86),
    ("appointed as", 0.88),
    ("honored by", 0.9),
    ("commemorated by", 0.9),
    ("recognized by", 0.88),
]
NAMED_AFTER_TRIGGER_SPECS = [
    ("named after", 0.9),
    ("in honor of", 0.9),
    ("in recognition of", 0.88),
]
AFFECTED_TRIGGER_SPECS = [
    ("was prosecuted for", 0.9),
    ("prosecuted for", 0.9),
    ("convicted of", 0.9),
    ("convicted for", 0.88),
    ("was pardoned", 0.9),
    ("pardoned", 0.88),
    ("affected by", 0.84),
    ("impacted by", 0.84),
]
DIRECT_TRIGGER_HYPOTHESIS_RELATIONS = {"EDUCATED_AT", "WORKED_AT", "PROPOSED"}
FIXED_HYPOTHESIS_VERBS = {
    "PARTICIPATED_IN": "participated in",
    "COLLEAGUE_OF": "worked with",
    "DIED_IN": "died in",
    "AFFECTED_BY": "was affected by",
    "HONORED_BY": "was honored by",
    "NAMED_AFTER": "was named after",
}
RULE_ASSIST_THRESHOLDS = {
    "WORKED_ON": (WORKED_ON_RULE_TRIGGERS, 0.88),
    "AUTHORED": (AUTHORED_RULE_TRIGGERS, 0.88),
    "COLLEAGUE_OF": (COLLEAGUE_RULE_TRIGGERS, 0.86),
    "PARTICIPATED_IN": (PARTICIPATED_RULE_TRIGGERS, 0.86),
    "HONORED_BY": (HONORED_RULE_TRIGGERS, 0.86),
    "NAMED_AFTER": (NAMED_AFTER_RULE_TRIGGERS, 0.86),
}
RULE_FIRST_OVERRIDES = {
    "WORKED_ON": (WORKED_ON_RULE_TRIGGERS.union({"led the development of", "helped develop"}), 0.9),
    "AUTHORED": (AUTHORED_RULE_TRIGGERS, 0.9),
    "COLLEAGUE_OF": (COLLEAGUE_RULE_TRIGGERS.union({"one of turing's lifelong friends and associates"}), 0.88),
    "PARTICIPATED_IN": (PARTICIPATED_RULE_TRIGGERS, 0.88),
    "HONORED_BY": (HONORED_RULE_TRIGGERS, 0.88),
    "NAMED_AFTER": (NAMED_AFTER_RULE_TRIGGERS, 0.88),
}
DEFAULT_PAIR_LIMIT = 2


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def normalize_entity_text(text: str) -> str:
    cleaned = text.strip().strip('"\'. ,;:()[]{}')
    return re.sub(r"\s+", " ", cleaned)


def is_center_entity(text: str) -> bool:
    normalized = normalize_entity_text(text).lower()
    return normalized in CENTER_ENTITY_NAMES or slugify(normalized) == CENTER_ENTITY_ID


def canonicalize_entity(entity: Dict) -> Dict:
    updated = dict(entity)
    text = normalize_entity_text(updated.get("text", ""))
    if is_center_entity(text):
        updated["text"] = CENTER_ENTITY_CANONICAL
    else:
        updated["text"] = text
    return updated


def normalize_page_entity_name(name: str) -> str:
    return normalize_entity_text(name).lower()


def resolve_page_subject_name(row: Dict) -> Optional[str]:
    for key in ("title", "seed_title"):
        value = row.get(key, "")
        normalized = normalize_page_entity_name(value)
        if normalized in PAGE_ENTITY_PRONOUNS:
            return value
    return None


def canonicalize_pronoun_entities(entities: List[Dict], row: Dict, sentence: str) -> List[Dict]:
    page_subject = resolve_page_subject_name(row)
    if not page_subject:
        return entities

    normalized_page = normalize_page_entity_name(page_subject)
    allowed_pronouns = PAGE_ENTITY_PRONOUNS.get(normalized_page, set())
    if not allowed_pronouns:
        return entities

    updated_entities: List[Dict] = []
    sentence_start = sentence.lstrip()
    for entity in entities:
        updated = dict(entity)
        text = normalize_entity_text(updated.get("text", ""))
        lowered = text.lower()
        if lowered in allowed_pronouns and updated.get("start", 9999) <= max(5, len(sentence) - len(sentence_start)):
            updated["text"] = page_subject
            updated["label"] = "Person"
        updated_entities.append(updated)
    return updated_entities


def is_valid_entity_text(text: str) -> bool:
    normalized = normalize_entity_text(text)
    if len(normalized) < 2:
        return False
    if normalized.lower() in {"and", "or", "the", "of", "for", "und"}:
        return False
    if re.fullmatch(r"[\W_]+", normalized):
        return False
    return True


def matches_allowed_type(label: str, allowed: Sequence[str]) -> bool:
    return label in allowed


def split_sentences(text: str) -> List[str]:
    split_pattern = re.compile(r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bProf)(?<=[.!?])\s+")
    return [segment.strip() for segment in split_pattern.split(text) if len(segment.strip()) > 10]


def relation_layer(start_text: str, end_text: str) -> str:
    return "core" if is_center_entity(start_text) or is_center_entity(end_text) else "expanded"


def threshold_for_rule_assist(relation: str, trigger_text: str) -> float:
    trigger = trigger_text.lower().strip()
    if relation in RULE_ASSIST_THRESHOLDS:
        trigger_set, value = RULE_ASSIST_THRESHOLDS[relation]
        if trigger in trigger_set:
            return value
    return 0.99


def dedupe_entities(entities: List[Dict]) -> List[Dict]:
    seen = set()
    cleaned = []
    for entity in entities:
        entity = canonicalize_entity(entity)
        text = entity.get("text", "")
        if not is_valid_entity_text(text):
            continue
        key = (entity.get("start"), entity.get("end"), entity.get("label"), text)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(entity)
    cleaned.sort(key=lambda item: (item["start"], item["end"]))
    return cleaned


def merge_adjacent_location(entities: List[Dict], start_idx: int) -> Tuple[str, int]:
    parts = [entities[start_idx]["text"]]
    end_idx = start_idx
    while end_idx + 1 < len(entities):
        current = entities[end_idx]
        nxt = entities[end_idx + 1]
        if current["label"] == "Location" and nxt["label"] == "Location" and nxt["start"] - current["end"] <= 3:
            parts.append(nxt["text"])
            end_idx += 1
        else:
            break
    return ", ".join(parts), end_idx


class TriggerCandidateGenerator:
    def __init__(self) -> None:
        self.compiled = {
            relation: [re.compile(pattern, re.IGNORECASE) for pattern in rules["triggers"]]
            for relation, rules in RELATION_SCHEMA.items()
        }

    def generate(self, sentence: str, entities: List[Dict]) -> List[Dict]:
        candidates: List[Dict] = []
        candidates.extend(self._education_rule_candidates(sentence, entities))
        candidates.extend(self._worked_on_rule_candidates(sentence, entities))
        candidates.extend(self._authored_rule_candidates(sentence, entities))
        candidates.extend(self._colleague_rule_candidates(sentence, entities))
        candidates.extend(self._participated_rule_candidates(sentence, entities))
        candidates.extend(self._affected_by_rule_candidates(sentence, entities))
        candidates.extend(self._honor_rule_candidates(sentence, entities))
        candidates.extend(self._named_after_rule_candidates(sentence, entities))
        for relation, rules in RELATION_SCHEMA.items():
            for pattern in self.compiled[relation]:
                for match in pattern.finditer(sentence):
                    candidates.extend(self._from_match(relation, rules, match, entities))
        return self._dedupe(candidates)

    def _append_phrase_candidate(
        self,
        candidates: List[Dict],
        sentence: str,
        entities: List[Dict],
        relation: str,
        phrase: str,
        allowed_sub: Sequence[str],
        allowed_obj: Sequence[str],
        rule_confidence: float,
        mode: str,
        subject_limit: int = 2,
        object_limit: int = 4,
        pair_limit: int = DEFAULT_PAIR_LIMIT,
        fallback_first_subject: Optional[List[int]] = None,
        object_search_from_phrase_end: bool = True,
    ) -> None:
        lowered = sentence.lower()
        pos = lowered.find(phrase)
        if pos == -1:
            return

        head_candidates = self._entities_before(entities, pos, allowed_sub, limit=subject_limit)
        if not head_candidates and fallback_first_subject:
            head_candidates = fallback_first_subject

        obj_pos = pos + len(phrase) if object_search_from_phrase_end else pos
        tail_candidates = self._entities_after(entities, obj_pos, allowed_obj, limit=object_limit)
        if not head_candidates or not tail_candidates:
            return

        for sub_idx, obj_idx in self._rank_pairs(head_candidates, tail_candidates, entities, pair_limit):
            candidates.append(
                {
                    "sub_idx": sub_idx,
                    "obj_idx": obj_idx,
                    "relation": relation,
                    "trigger_text": phrase,
                    "rule_confidence": rule_confidence,
                    "mode": mode,
                    "nli_hypothesis": self._build_hypothesis_from_trigger(
                        relation,
                        phrase,
                        entities[sub_idx],
                        entities[obj_idx],
                        entities,
                    ),
                }
            )

    def _append_first_pair_candidate(
        self,
        candidates: List[Dict],
        entities: List[Dict],
        relation: str,
        trigger_text: str,
        confidence: float,
        left_candidates: List[int],
        right_candidates: List[int],
        fallback_pair: Optional[Tuple[int, int]] = None,
    ) -> None:
        if left_candidates and right_candidates:
            sub_idx, obj_idx = left_candidates[0], right_candidates[0]
        elif fallback_pair is not None:
            sub_idx, obj_idx = fallback_pair
        else:
            return

        candidates.append(
            {
                "sub_idx": sub_idx,
                "obj_idx": obj_idx,
                "relation": relation,
                "trigger_text": trigger_text,
                "rule_confidence": confidence,
                "mode": "rule_first" if confidence >= 0.88 else "nli_assist",
                "nli_hypothesis": self._build_hypothesis_from_trigger(
                    relation,
                    trigger_text,
                    entities[sub_idx],
                    entities[obj_idx],
                    entities,
                ),
            }
        )

    def _append_phrase_text_candidate(
        self,
        candidates: List[Dict],
        entities: List[Dict],
        relation: str,
        trigger_text: str,
        confidence: float,
        subject_idx: int,
        object_text: str,
        object_label: str,
    ) -> None:
        object_text = normalize_entity_text(object_text)
        if not object_text:
            return
        head = entities[subject_idx]
        candidates.append(
            {
                "sub_idx": subject_idx,
                "obj_idx": None,
                "relation": relation,
                "trigger_text": trigger_text,
                "rule_confidence": confidence,
                "mode": "rule_first" if confidence >= 0.88 else "nli_assist",
                "obj_text": object_text,
                "obj_label": object_label,
                "nli_hypothesis": self._build_hypothesis_from_text(
                    relation,
                    trigger_text,
                    normalize_entity_text(head["text"]),
                    object_text,
                ),
            }
        )

    def _find_phrase(self, sentence: str, phrase: str) -> int:
        return sentence.lower().find(phrase)

    def _tail_phrase(
        self,
        sentence: str,
        start_pos: int,
        stop_words: Sequence[str],
    ) -> str:
        tail = sentence[start_pos:].strip(" ,.;:-")
        if not tail:
            return ""
        cut = len(tail)
        lowered = tail.lower()
        for stop_word in stop_words:
            match = re.search(rf"\b{re.escape(stop_word)}\b", lowered)
            if match:
                cut = min(cut, match.start())
        return normalize_entity_text(tail[:cut].strip(" ,.;:-"))

    def _rank_pairs(
        self,
        left: List[int],
        right: List[int],
        entities: List[Dict],
        pair_limit: int,
    ) -> List[Tuple[int, int]]:
        ranked: List[Tuple[int, int, int]] = []
        for left_idx in left:
            for right_idx in right:
                if left_idx == right_idx:
                    continue
                distance = max(0, entities[right_idx]["start"] - entities[left_idx]["end"])
                ranked.append((distance, left_idx, right_idx))
        ranked.sort(key=lambda item: item[0])
        return [(left_idx, right_idx) for _, left_idx, right_idx in ranked[:pair_limit]]

    def _from_match(self, relation: str, rules: Dict, match: re.Match, entities: List[Dict]) -> List[Dict]:
        trigger_text = match.group(0)
        trigger_span = match.span()
        strategy = rules["strategy"]
        head_candidates = self._entities_before(entities, trigger_span[0], rules["sub"])
        tail_candidates = self._entities_after(entities, trigger_span[1], rules["obj"])

        if strategy == "subject_before_object_after":
            return self._pair_nearest(head_candidates, tail_candidates, relation, trigger_text, entities, pair_limit=DEFAULT_PAIR_LIMIT)
        if strategy == "person_pair":
            left_people = self._entities_before(entities, trigger_span[0], rules["sub"], limit=2)
            right_people = self._entities_after(entities, trigger_span[1], rules["obj"], limit=2)
            return self._pair_nearest(left_people, right_people, relation, trigger_text, entities, pair_limit=DEFAULT_PAIR_LIMIT)
        if strategy == "subject_before_object_after_or_event_before":
            direct = self._pair_nearest(head_candidates, tail_candidates, relation, trigger_text, entities, pair_limit=DEFAULT_PAIR_LIMIT)
            if direct:
                return direct
            event_before = self._entities_before(entities, trigger_span[0], rules["obj"], limit=2)
            subject_after = self._entities_after(entities, trigger_span[1], rules["sub"], limit=2)
            return [
                self._candidate(sub_idx, evt_idx, relation, trigger_text, entities)
                for evt_idx in event_before for sub_idx in subject_after
            ][:DEFAULT_PAIR_LIMIT]
        if strategy == "subject_before_object_after_or_object_before":
            direct = self._pair_nearest(head_candidates, tail_candidates, relation, trigger_text, entities, pair_limit=DEFAULT_PAIR_LIMIT)
            if direct:
                return direct
            object_before = self._entities_before(entities, trigger_span[0], rules["obj"], limit=2)
            subject_after = self._entities_after(entities, trigger_span[1], rules["sub"], limit=2)
            return [
                self._candidate(sub_idx, obj_idx, relation, trigger_text, entities)
                for obj_idx in object_before for sub_idx in subject_after
            ][:DEFAULT_PAIR_LIMIT]
        return []

    def _education_rule_candidates(self, sentence: str, entities: List[Dict]) -> List[Dict]:
        lowered = sentence.lower()
        candidates: List[Dict] = []
        person_indices = [idx for idx, entity in enumerate(entities) if entity["label"] == "Person"]
        org_indices = [idx for idx, entity in enumerate(entities) if entity["label"] == "Organization"]
        if not person_indices or not org_indices:
            return candidates

        for phrase in EDUCATION_PHRASE_HINTS:
            self._append_phrase_candidate(
                candidates,
                sentence,
                entities,
                "EDUCATED_AT",
                phrase,
                ["Person"],
                ["Organization"],
                0.98,
                "rule_first",
            )

        for org_idx in org_indices:
            org_text = normalize_entity_text(entities[org_idx]["text"]).lower()
            if org_text not in EDUCATION_ORG_HINTS:
                continue
            for person_idx in person_indices:
                if entities[person_idx]["end"] <= entities[org_idx]["start"]:
                    window = lowered[max(0, entities[person_idx]["start"] - 40): min(len(lowered), entities[org_idx]["end"] + 40)]
                    if any(phrase in window for phrase in EDUCATION_PHRASE_HINTS):
                        candidates.append(
                            {
                                "sub_idx": person_idx,
                                "obj_idx": org_idx,
                                "relation": "EDUCATED_AT",
                                "trigger_text": "education_org_hint",
                                "rule_confidence": 0.99,
                                "mode": "rule_first",
                                "nli_hypothesis": self._build_hypothesis_from_trigger(
                                    "EDUCATED_AT",
                                    "studied at",
                                    entities[person_idx],
                                    entities[org_idx],
                                    entities,
                                ),
                            }
                        )

        return candidates

    def _worked_on_rule_candidates(self, sentence: str, entities: List[Dict]) -> List[Dict]:
        candidates: List[Dict] = []
        person_indices = [idx for idx, entity in enumerate(entities) if entity["label"] == "Person"]
        obj_indices = [
            idx for idx, entity in enumerate(entities)
            if entity["label"] in {"Artifact", "Concept"}
        ]
        if not person_indices or not obj_indices:
            return candidates

        for phrase, confidence in WORKED_ON_TRIGGER_SPECS:
            self._append_phrase_candidate(
                candidates,
                sentence,
                entities,
                "WORKED_ON",
                phrase,
                ["Person"],
                ["Artifact", "Concept"],
                confidence,
                "rule_first" if confidence >= 0.88 else "nli_assist",
            )
        return candidates

    def _authored_rule_candidates(self, sentence: str, entities: List[Dict]) -> List[Dict]:
        candidates: List[Dict] = []
        person_indices = [idx for idx, entity in enumerate(entities) if entity["label"] == "Person"]
        pub_indices = [idx for idx, entity in enumerate(entities) if entity["label"] == "Publication"]
        if not person_indices or not pub_indices:
            return candidates

        for phrase, confidence in AUTHORED_TRIGGER_SPECS:
            self._append_phrase_candidate(
                candidates,
                sentence,
                entities,
                "AUTHORED",
                phrase,
                ["Person"],
                ["Publication"],
                confidence,
                "rule_first" if confidence >= 0.88 else "nli_assist",
                fallback_first_subject=person_indices[:1] if phrase in {"in his paper", "in her paper", "in his 1936 paper", "in her 1936 paper"} else None,
            )
        return candidates

    def _colleague_rule_candidates(self, sentence: str, entities: List[Dict]) -> List[Dict]:
        candidates: List[Dict] = []
        person_indices = [idx for idx, entity in enumerate(entities) if entity["label"] == "Person"]
        if len(person_indices) < 2:
            return candidates

        fallback_phrases = {"lifelong friends and associates", "friends and associates"}
        for phrase, confidence in COLLEAGUE_TRIGGER_SPECS:
            pos = self._find_phrase(sentence, phrase)
            if pos == -1:
                continue
            left_people = self._entities_before(entities, pos, ["Person"], limit=2)
            right_people = self._entities_after(entities, pos + len(phrase), ["Person"], limit=2)
            fallback_pair = (person_indices[0], person_indices[1]) if phrase in fallback_phrases else None
            self._append_first_pair_candidate(
                candidates,
                entities,
                "COLLEAGUE_OF",
                phrase,
                confidence,
                left_people,
                right_people,
                fallback_pair=fallback_pair,
            )
        return candidates

    def _participated_rule_candidates(self, sentence: str, entities: List[Dict]) -> List[Dict]:
        candidates: List[Dict] = []
        event_indices = [idx for idx, entity in enumerate(entities) if entity["label"] == "Event"]
        if not event_indices:
            return candidates

        person_or_org = [
            idx for idx, entity in enumerate(entities)
            if entity["label"] in {"Person", "Organization"}
        ]
        if not person_or_org:
            return candidates

        for phrase, confidence in PARTICIPATED_TRIGGER_SPECS:
            pos = self._find_phrase(sentence, phrase)
            if pos == -1:
                continue
            event_before = self._entities_before(entities, pos + len(phrase), ["Event"], limit=2)
            event_after = self._entities_after(entities, pos, ["Event"], limit=2)
            event_candidates = event_before or event_after or event_indices[:1]
            subject_before = self._entities_before(entities, pos, ["Person", "Organization"], limit=2)
            subject_after = self._entities_after(entities, pos + len(phrase), ["Person", "Organization"], limit=2)
            subject_candidates = subject_before or subject_after or person_or_org[:1]
            self._append_first_pair_candidate(
                candidates,
                entities,
                "PARTICIPATED_IN",
                phrase,
                confidence,
                subject_candidates,
                event_candidates,
            )
        return candidates

    def _honor_rule_candidates(self, sentence: str, entities: List[Dict]) -> List[Dict]:
        candidates: List[Dict] = []
        person_indices = [idx for idx, entity in enumerate(entities) if entity["label"] == "Person"]
        if not person_indices:
            return candidates

        for phrase, confidence in HONOR_TRIGGER_SPECS:
            pos = self._find_phrase(sentence, phrase)
            if pos == -1:
                continue
            person_before = self._entities_before(entities, pos, ["Person"], limit=2)
            honor_after = self._entities_after(entities, pos + len(phrase), list(HONORED_OBJECT_LABELS), limit=3)
            self._append_first_pair_candidate(
                candidates,
                entities,
                "HONORED_BY",
                phrase,
                confidence,
                person_before,
                honor_after,
            )
            if person_before:
                honor_text = self._tail_phrase(
                    sentence,
                    pos + len(phrase),
                    stop_words=("who", "that", "which", "during", "in", "on", "at", "for"),
                )
                if honor_text:
                    self._append_phrase_text_candidate(
                        candidates,
                        entities,
                        "HONORED_BY",
                        phrase,
                        max(confidence, 0.88 if phrase in {"appointed as", "honored by", "commemorated by"} else confidence),
                        person_before[0],
                        honor_text,
                        "Honor",
                    )
        return candidates

    def _named_after_rule_candidates(self, sentence: str, entities: List[Dict]) -> List[Dict]:
        candidates: List[Dict] = []
        subject_indices = [
            idx for idx, entity in enumerate(entities)
            if entity["label"] in NAMED_AFTER_SUBJECT_LABELS
        ]
        person_indices = [idx for idx, entity in enumerate(entities) if entity["label"] == "Person"]
        if not subject_indices or not person_indices:
            return candidates

        for phrase, confidence in NAMED_AFTER_TRIGGER_SPECS:
            pos = self._find_phrase(sentence, phrase)
            if pos == -1:
                continue
            subject_before = self._entities_before(entities, pos, ["Honor", "Organization", "Artifact"], limit=2)
            person_after = self._entities_after(entities, pos + len(phrase), ["Person"], limit=2)
            self._append_first_pair_candidate(
                candidates,
                entities,
                "NAMED_AFTER",
                phrase,
                confidence,
                subject_before,
                person_after,
            )
            if person_after:
                subject_text = sentence[:pos].strip(" ,.;:-")
                if subject_text:
                    fallback_subject = subject_before[:1] or subject_indices[:1]
                    if fallback_subject:
                        self._append_first_pair_candidate(
                            candidates,
                            entities,
                            "NAMED_AFTER",
                            phrase,
                            confidence,
                            fallback_subject,
                            person_after,
                        )
        return candidates

    def _affected_by_rule_candidates(self, sentence: str, entities: List[Dict]) -> List[Dict]:
        candidates: List[Dict] = []
        person_indices = [idx for idx, entity in enumerate(entities) if entity["label"] == "Person"]
        if not person_indices:
            return candidates

        for phrase, confidence in AFFECTED_TRIGGER_SPECS:
            pos = self._find_phrase(sentence, phrase)
            if pos == -1:
                continue
            person_before = self._entities_before(entities, pos, ["Person"], limit=2)
            object_after = self._entities_after(entities, pos + len(phrase), list(AFFECTED_OBJECT_LABELS), limit=3)
            self._append_first_pair_candidate(
                candidates,
                entities,
                "AFFECTED_BY",
                phrase,
                confidence,
                person_before,
                object_after,
            )
            if person_before:
                object_text = self._tail_phrase(
                    sentence,
                    pos + len(phrase),
                    stop_words=("after", "before", "when", "because", "and", "but", "which", "that"),
                )
                if object_text:
                    self._append_phrase_text_candidate(
                        candidates,
                        entities,
                        "AFFECTED_BY",
                        phrase,
                        confidence,
                        person_before[0],
                        object_text,
                        "Event",
                    )
        return candidates

    def _entities_before(self, entities: List[Dict], pos: int, allowed: Sequence[str], limit: int = 3) -> List[int]:
        matches = [
            idx for idx, entity in enumerate(entities)
            if entity["end"] <= pos and matches_allowed_type(entity["label"], allowed)
        ]
        matches.sort(key=lambda idx: pos - entities[idx]["end"])
        return matches[:limit]

    def _entities_after(self, entities: List[Dict], pos: int, allowed: Sequence[str], limit: int = 4) -> List[int]:
        matches = [
            idx for idx, entity in enumerate(entities)
            if entity["start"] >= pos and matches_allowed_type(entity["label"], allowed)
        ]
        matches.sort(key=lambda idx: entities[idx]["start"] - pos)
        return matches[:limit]

    def _candidate(self, sub_idx: int, obj_idx: int, relation: str, trigger_text: str, entities: List[Dict]) -> Dict:
        mode, rule_confidence = self._override_mode_and_confidence(relation, trigger_text)
        candidate = {
            "sub_idx": sub_idx,
            "obj_idx": obj_idx,
            "relation": relation,
            "trigger_text": trigger_text,
            "rule_confidence": rule_confidence,
            "mode": mode,
            "nli_hypothesis": self._build_hypothesis_from_trigger(relation, trigger_text, entities[sub_idx], entities[obj_idx], entities),
        }
        return candidate

    def _build_hypothesis_from_text(self, relation: str, trigger_text: str, head_text: str, tail_text: str) -> str:
        if relation == "EDUCATED_AT" and trigger_text == "education_org_hint":
            return f"{head_text} studied at {tail_text}."
        if relation in DIRECT_TRIGGER_HYPOTHESIS_RELATIONS:
            return f"{head_text} {trigger_text} {tail_text}."
        if relation == "WORKED_ON":
            if trigger_text in {"designed", "developed", "improved", "devised", "led the development of", "helped develop"}:
                return f"{head_text} {trigger_text} {tail_text}."
            return f"{head_text} worked on {tail_text}."
        if relation == "AUTHORED":
            if trigger_text in AUTHORED_RULE_TRIGGERS.union({"published"}):
                return f"{head_text} {trigger_text} {tail_text}."
            return f"{head_text} authored {tail_text}."
        if relation in FIXED_HYPOTHESIS_VERBS:
            return f"{head_text} {FIXED_HYPOTHESIS_VERBS[relation]} {tail_text}."
        return f"{head_text} {trigger_text} {tail_text}."

    def _pair_nearest(
        self,
        left: List[int],
        right: List[int],
        relation: str,
        trigger_text: str,
        entities: List[Dict],
        pair_limit: int = DEFAULT_PAIR_LIMIT,
    ) -> List[Dict]:
        if not left or not right:
            return []
        return [
            self._candidate(sub_idx, obj_idx, relation, trigger_text, entities)
            for sub_idx, obj_idx in self._rank_pairs(left, right, entities, pair_limit)
        ]

    def _dedupe(self, candidates: List[Dict]) -> List[Dict]:
        seen = set()
        ordered = []
        for candidate in candidates:
            key = (candidate["sub_idx"], candidate["obj_idx"], candidate["relation"], candidate["trigger_text"])
            if key in seen:
                continue
            seen.add(key)
            ordered.append(candidate)
        return ordered

    def _override_mode_and_confidence(self, relation: str, trigger_text: str) -> Tuple[str, float]:
        trigger = trigger_text.lower().strip()
        default_mode = RELATION_SCHEMA[relation]["mode"]
        default_conf = RELATION_SCHEMA[relation]["rule_confidence"]
        if relation in RULE_FIRST_OVERRIDES:
            trigger_set, confidence = RULE_FIRST_OVERRIDES[relation]
            if trigger in trigger_set:
                return "rule_first", confidence
        return default_mode, default_conf

    def _build_hypothesis_from_trigger(self, relation: str, trigger_text: str, head: Dict, tail: Dict, entities: List[Dict]) -> str:
        head_text = normalize_entity_text(head["text"])
        tail_text = normalize_entity_text(tail["text"])
        trigger = trigger_text.strip()

        if relation == "BORN_IN" and tail["label"] == "Location":
            try:
                idx = entities.index(tail)
            except ValueError:
                idx = -1
            if idx >= 0:
                merged, _ = merge_adjacent_location(entities, idx)
                tail_text = merged
            return f"{head_text} was born in {tail_text}."
        return self._build_hypothesis_from_text(relation, trigger, head_text, tail_text)


class RuleFirstRelationExtractor:
    def __init__(self, model_name: str = "cross-encoder/nli-distilroberta-base", device: str = "cpu", local_files_only: bool = True):
        self.device = device
        self.generator = TriggerCandidateGenerator()
        self.tokenizer = None
        self.model = None
        if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
            print(f"Loading NLI Model: {model_name} on {device}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    local_files_only=local_files_only,
                ).to(device)
            except Exception as exc:
                print(f"[WARN] NLI model unavailable, falling back to rule-only mode: {exc}")
                self.tokenizer = None
                self.model = None

    def score_candidates(self, sentence: str, entities: List[Dict]) -> List[Dict]:
        candidates = self.generator.generate(sentence, entities)
        if not candidates:
            return []

        nli_candidates = [c for c in candidates if c["mode"] != "rule_first"]
        nli_scores: List[float] = []
        if nli_candidates and self.model is not None and self.tokenizer is not None and TORCH_AVAILABLE:
            inputs = self.tokenizer(
                [sentence for _ in nli_candidates],
                [c["nli_hypothesis"] for c in nli_candidates],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            nli_scores = [float(probs[idx][2].item()) for idx in range(len(nli_candidates))]

        nli_index = 0
        scored: List[Dict] = []
        for candidate in candidates:
            head = entities[candidate["sub_idx"]]
            tail = entities[candidate["obj_idx"]] if candidate["obj_idx"] is not None else None
            end_text = normalize_entity_text(tail["text"]) if tail is not None else candidate.get("obj_text", "")
            end_label = tail["label"] if tail is not None else candidate.get("obj_label", "Concept")
            record = {
                "sub_idx": candidate["sub_idx"],
                "obj_idx": candidate["obj_idx"],
                "relation": candidate["relation"],
                "trigger_text": candidate["trigger_text"],
                "mode": candidate["mode"],
                "hypothesis": candidate["nli_hypothesis"],
                "start_text": normalize_entity_text(head["text"]),
                "start_label": head["label"],
                "end_text": end_text,
                "end_label": end_label,
                "rule_confidence": candidate["rule_confidence"],
                "nli_score": None,
            }
            if candidate["mode"] == "rule_first":
                record["final_score"] = candidate["rule_confidence"]
                record["passed"] = True
                record["decision_source"] = "rule"
            else:
                nli_score = nli_scores[nli_index] if nli_index < len(nli_scores) else 0.0
                nli_index += 1
                nli_threshold = NLI_ASSIST_THRESHOLDS.get(candidate["relation"], 0.45)
                record["nli_score"] = round(nli_score, 6)
                record["final_score"] = max(candidate["rule_confidence"], nli_score)
                record["passed"] = (nli_score >= nli_threshold) or (candidate["rule_confidence"] >= threshold_for_rule_assist(candidate["relation"], candidate["trigger_text"]))
                record["decision_source"] = "nli" if nli_score >= nli_threshold else "rule"
            scored.append(record)
        return scored

    def extract_relations_batch(self, sentence: str, entities: List[Dict], threshold: float) -> List[Dict]:
        scored = self.score_candidates(sentence, entities)
        if not scored:
            return []

        best_results: Dict[Tuple[int, int, str], Dict] = {}
        for item in scored:
            if item["mode"] != "rule_first" and item["final_score"] < threshold:
                continue
            if item["mode"] != "rule_first" and not item["passed"]:
                continue

            key = (item["sub_idx"], item["obj_idx"] if item["obj_idx"] is not None else item["end_text"], item["relation"])
            if key not in best_results or item["final_score"] > best_results[key]["final_score"]:
                best_results[key] = item

        extracted = []
        for (_, _, relation), item in best_results.items():
            start_text = item["start_text"]
            end_text = item["end_text"]
            if item["start_label"] == item["end_label"] == "Person" and start_text == end_text:
                continue
            layer = relation_layer(start_text, end_text)
            extracted.append(
                {
                    "relation": relation,
                    "start_text": start_text,
                    "start_label": item["start_label"],
                    "end_text": end_text,
                    "end_label": item["end_label"],
                    "start_uid": f"{item['start_label']}:{slugify(start_text)}",
                    "end_uid": f"{item['end_label']}:{slugify(end_text)}",
                    "confidence": round(float(item["final_score"]), 4),
                    "layer": layer,
                    "centered_on": CENTER_ENTITY_ID,
                    "is_core": layer == "core",
                    "decision_source": item["decision_source"],
                    "trigger_text": item["trigger_text"],
                }
            )
        return extracted


def keep_centered_relations(records: List[Dict], expand_threshold: float) -> List[Dict]:
    one_hop_ids: Set[str] = set()
    for record in records:
        if record["start_uid"].endswith(f":{CENTER_ENTITY_ID}") or record["end_uid"].endswith(f":{CENTER_ENTITY_ID}"):
            one_hop_ids.add(record["start_uid"])
            one_hop_ids.add(record["end_uid"])

    filtered: List[Dict] = []
    seen = set()
    for record in records:
        start_is_center = record["start_uid"].endswith(f":{CENTER_ENTITY_ID}")
        end_is_center = record["end_uid"].endswith(f":{CENTER_ENTITY_ID}")
        if start_is_center or end_is_center:
            key = (record["start_uid"], record["relation"], record["end_uid"], record["sent_id"])
            if key not in seen:
                seen.add(key)
                filtered.append(record)
            continue

        if record["confidence"] < expand_threshold:
            continue
        if record["start_uid"] not in one_hop_ids and record["end_uid"] not in one_hop_ids:
            continue
        if record["layer"] != "expanded":
            continue

        key = (record["start_uid"], record["relation"], record["end_uid"], record["sent_id"])
        if key not in seen:
            seen.add(key)
            filtered.append(record)

    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/turing_schema_corpus.jsonl")
    parser.add_argument("--ner-dir", default="data/output/ner_bilstm_crf")
    parser.add_argument("--output", default="data/output/relations_dev.jsonl")
    parser.add_argument("--bert-model", default="cross-encoder/nli-distilroberta-base")
    parser.add_argument("--allow-remote-model", action="store_true", help="Allow downloading the NLI model if it is not cached locally.")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--expand-threshold", type=float, default=0.82)
    args = parser.parse_args()

    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        raise SystemExit("Relation extraction requires 'torch' and 'transformers' to be installed.")

    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner_model, cv, tv = ner_module.load_ner_model_standalone(Path(args.ner_dir), model_device)
    extractor = RuleFirstRelationExtractor(
        args.bert_model,
        model_device,
        local_files_only=not args.allow_remote_model,
    )

    with open(args.input, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    all_relations: List[Dict] = []

    for row in tqdm(rows, desc="Extracting Relations"):
        text = row.get("text", "")
        doc_id = str(row.get("doc_id", row.get("id", "doc")))
        source = str(row.get("url", "")) or doc_id

        for sent_idx, sent_text in enumerate(split_sentences(text)):
            entities = ner_module.predict_entities(ner_model, cv, tv, sent_text, model_device)
            entities = canonicalize_pronoun_entities(entities, row, sent_text)
            entities = dedupe_entities(entities)
            if len(entities) < 2:
                continue

            scored_candidates = extractor.score_candidates(sent_text, entities)
            relations = extractor.extract_relations_batch(sent_text, entities, args.threshold)
            for relation in relations:
                relation["doc_id"] = doc_id
                relation["sent_id"] = f"{doc_id}_s{sent_idx:03d}"
                relation["evidence"] = sent_text
                relation["source"] = source
                relation["method"] = "rule_first_plus_nli"
                all_relations.append(relation)

    final_relations = keep_centered_relations(all_relations, args.expand_threshold)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for record in final_relations:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[DONE] Extracted {len(final_relations)} centered relations -> {args.output}")


if __name__ == "__main__":
    main()
