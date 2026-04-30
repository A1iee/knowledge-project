import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModelForSequenceClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    TRANSFORMERS_AVAILABLE = False

from tqdm import tqdm

try:
    import train_ner_bilstm_crf as ner_module
except ImportError:
    import scripts.train_ner_bilstm_crf as ner_module


ALLOWED_ATTRIBUTES = {
    "Person": ["birth_date", "death_date", "nationality", "occupation", "aliases"],
    "Organization": ["established_year", "aliases", "org_type", "location"],
    "Publication": ["publish_year", "venue", "aliases"],
    "Event": ["start_date", "end_date"],
    "Artifact": ["creation_year", "aliases", "description"],
    "Concept": ["creation_year", "aliases", "description"],
}

CENTER_PERSON_ALIASES = {
    "turing": "Alan Turing",
    "alan turing": "Alan Turing",
    "alan mathison turing": "Alan Turing",
    "he": "Alan Turing",
    "clarke": "Joan Clarke",
    "joan clarke": "Joan Clarke",
    "she": "Joan Clarke",
}
PSEUDO_PERSON_UPPERCASE = {"MBE", "ICS", "GCHQ", "NPL", "GC&CS"}
NATIONALITIES = {
    "British", "English", "American", "German", "French", "Russian",
    "Polish", "Hungarian", "Dutch", "Italian", "Austrian", "Canadian", "Scottish",
}
OCCUPATIONS = {
    "mathematician", "logician", "cryptanalyst", "philosopher", "biologist",
    "scientist", "engineer", "professor", "programmer", "codebreaker",
    "researcher", "author", "physicist", "inventor", "computer scientist",
}
ORG_TYPES = {
    "university", "college", "laboratory", "school", "park", "institute", "office", "government",
}
VENUE_HINTS = {
    "journal", "conference", "proceedings", "press", "review", "transactions",
}
ATTRIBUTE_NLI_THRESHOLDS = {
    "birth_date": 0.62,
    "death_date": 0.62,
    "nationality": 0.58,
    "occupation": 0.56,
    "aliases": 0.6,
    "established_year": 0.62,
    "org_type": 0.62,
    "location": 0.56,
    "publish_year": 0.62,
    "venue": 0.58,
    "start_date": 0.62,
    "end_date": 0.62,
    "creation_year": 0.62,
    "description": 0.58,
}


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip('"\'. ,;:()[]{}'))


def canonicalize_entity_text(text: str, label: str) -> str:
    normalized = normalize_text(text)
    if label == "Person":
        lowered = normalized.lower()
        if lowered in CENTER_PERSON_ALIASES:
            return CENTER_PERSON_ALIASES[lowered]
    return normalized


def is_valid_attribute_entity(entity: Dict[str, str]) -> bool:
    label = entity.get("label", "")
    text = normalize_text(entity.get("text", ""))
    if not text:
        return False
    if label == "Person":
        if text.isupper() and text in PSEUDO_PERSON_UPPERCASE:
            return False
        if len(text) <= 4 and text.isupper():
            return False
    return True


def title_from_doc_id(doc_id: str) -> str:
    match = re.match(r"^wiki_(.+?)_\d+$", doc_id or "")
    if not match:
        return ""
    raw = match.group(1).replace("_", " ").strip()
    return " ".join(part.capitalize() if part.islower() else part for part in raw.split())


def infer_page_subject_entity(doc_id: str) -> Optional[Dict[str, str]]:
    title = title_from_doc_id(doc_id)
    if not title:
        return None
    if title in {"Alan Turing", "Joan Clarke", "Alonzo Church", "Gordon Welchman", "John von Neumann"}:
        return {"text": title, "label": "Person"}
    if any(keyword in title for keyword in ["University", "College", "Laboratory", "School", "Park", "Institute"]):
        return {"text": title, "label": "Organization"}
    return None


def split_sentences(text: str) -> List[str]:
    split_pattern = re.compile(r"(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\ba\.k\.a)(?<=[.!?])\s+")
    return [s.strip() for s in split_pattern.split(text) if len(s.strip()) > 10]


class TypedCandidateExtractor:
    def __init__(self) -> None:
        month = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        self.full_date_pattern = re.compile(rf"\b(?:[0-3]?\d\s+{month}\s+\d{{4}}|{month}\s+[0-3]?\d,?\s+\d{{4}})\b", re.IGNORECASE)
        self.year_pattern = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
        self.alias_pattern = re.compile(r"\b(?:also known as|a\.k\.a\.|alias|referred to as|known as)\s+([A-Z][A-Za-z0-9 .'\-]{1,80})", re.IGNORECASE)
        self.description_pattern = re.compile(r"\b(?:is|was|refers to|describes)\s+(?:an?|the)\s+([^.;]{5,120})", re.IGNORECASE)
        self.location_pattern = re.compile(r"\b(?:in|at|from|based in|located in)\s+([A-Z][A-Za-z .,'\-]{1,80})")
        self.venue_pattern = re.compile(r"\b(?:in|at|published in)\s+([A-Z][A-Za-z0-9 .,'&\-]{1,80})")
        self.birth_context_patterns = [
            re.compile(rf"\b(?:{month}\s+[0-3]?\d,?\s+\d{{4}}|[0-3]?\d\s+{month}\s+\d{{4}})\b(?=[^.!?]{{0,80}}\b(?:birth|born)\b)", re.IGNORECASE),
            re.compile(rf"\b(?:born|birth)\b[^.!?]{{0,80}}\b([0-3]?\d\s+{month}\s+\d{{4}}|{month}\s+[0-3]?\d,?\s+\d{{4}})\b", re.IGNORECASE),
        ]
        self.death_context_patterns = [
            re.compile(rf"\b(?:{month}\s+[0-3]?\d,?\s+\d{{4}}|[0-3]?\d\s+{month}\s+\d{{4}})\b(?=[^.!?]{{0,80}}\b(?:died|death|passed away)\b)", re.IGNORECASE),
            re.compile(rf"\b(?:died|death|passed away)\b[^.!?]{{0,80}}\b([0-3]?\d\s+{month}\s+\d{{4}}|{month}\s+[0-3]?\d,?\s+\d{{4}})\b", re.IGNORECASE),
        ]
        self.lead_occupation_pattern = re.compile(
            r"\b(?:is|was)\s+(?:an?|the)\s+([a-z][a-z ,\-]{2,160})",
            re.IGNORECASE,
        )
        self.parenthetical_alias_pattern = re.compile(
            r"\b([A-Z][A-Za-z .'\-]{1,80})\s*\(([^()]{2,60})\)"
        )
        self.quoted_alias_pattern = re.compile(
            r"\b(?:also known as|known as|called)\s+[\"“]([^\"”]{2,60})[\"”]",
            re.IGNORECASE,
        )

    def person_reference_forms(self, entity_text: str) -> Set[str]:
        normalized = normalize_text(entity_text)
        if not normalized:
            return set()
        parts = normalized.split()
        forms = {normalized}
        if parts:
            forms.add(parts[-1])
            forms.add(parts[0])
        if normalized == "Alan Turing":
            forms.update({"Turing", "He", "His"})
        if normalized == "Joan Clarke":
            forms.update({"Clarke", "She", "Her"})
        return {form for form in forms if form}

    def sentence_mentions_person(self, sentence: str, entity_text: str) -> bool:
        sentence_norm = normalize_text(sentence)
        for form in self.person_reference_forms(entity_text):
            if re.search(rf"\b{re.escape(form)}\b", sentence_norm, re.IGNORECASE):
                return True
        return False

    def extract(self, sentence: str, entity: Dict[str, str]) -> Dict[str, List[str]]:
        label = entity["label"]
        entity_text = canonicalize_entity_text(entity["text"], label)
        allowed = ALLOWED_ATTRIBUTES.get(label, [])
        candidates: Dict[str, List[str]] = {attr: [] for attr in allowed}

        if label == "Person":
            candidates["birth_date"] = self.person_birth_dates(sentence, entity_text)
            candidates["death_date"] = self.person_death_dates(sentence, entity_text)
            candidates["nationality"] = self.person_nationalities(sentence, entity_text)
            candidates["occupation"] = self.person_occupations(sentence, entity_text)
            candidates["aliases"] = self.person_aliases(sentence, entity_text)
        elif label == "Organization":
            candidates["established_year"] = self.years(sentence)
            candidates["aliases"] = self.aliases(sentence)
            candidates["org_type"] = [org_type for org_type in ORG_TYPES if re.search(rf"\b{re.escape(org_type)}\b", sentence, re.IGNORECASE)]
            candidates["location"] = self.locations(sentence)
        elif label == "Publication":
            candidates["publish_year"] = self.years(sentence)
            candidates["venue"] = self.venues(sentence)
            candidates["aliases"] = self.aliases(sentence)
        elif label == "Event":
            dates = self.full_dates(sentence) or self.years(sentence)
            candidates["start_date"] = dates
            candidates["end_date"] = dates
        elif label in {"Artifact", "Concept"}:
            candidates["creation_year"] = self.years(sentence)
            candidates["aliases"] = self.aliases(sentence)
            description = self.description(sentence)
            if description:
                candidates["description"] = [description]

        return {key: list(dict.fromkeys(values)) for key, values in candidates.items() if values}

    def person_nationalities(self, sentence: str, entity_text: str) -> List[str]:
        candidates: List[str] = []
        sentence_norm = normalize_text(sentence)
        entity_pattern = re.escape(entity_text)
        for nationality in NATIONALITIES:
            if re.search(rf"\b{re.escape(nationality)}\s+[A-Za-z -]{{0,30}}{entity_pattern}\b", sentence_norm, re.IGNORECASE):
                candidates.append(nationality)
            if re.search(rf"\b{entity_pattern}\b\s+(?:was|is)\s+an?\s+{re.escape(nationality)}\b", sentence_norm, re.IGNORECASE):
                candidates.append(nationality)
            if re.search(rf"\b{entity_pattern}\b\s*,\s+an?\s+{re.escape(nationality)}\b", sentence_norm, re.IGNORECASE):
                candidates.append(nationality)
        return list(dict.fromkeys(candidates))

    def person_birth_dates(self, sentence: str, entity_text: str) -> List[str]:
        if not self.sentence_mentions_person(sentence, entity_text):
            return []
        values: List[str] = []
        for pattern in self.birth_context_patterns:
            for match in pattern.findall(sentence):
                value = match if isinstance(match, str) else match[0]
                normalized = normalize_text(value)
                if normalized:
                    values.append(normalized)
        if values:
            return list(dict.fromkeys(values))
        if re.search(rf"\b{re.escape(entity_text)}\b[^.!?]{{0,80}}\bwas born on\b", sentence, re.IGNORECASE):
            return self.full_dates(sentence)
        return []

    def person_death_dates(self, sentence: str, entity_text: str) -> List[str]:
        if not self.sentence_mentions_person(sentence, entity_text):
            return []
        values: List[str] = []
        for pattern in self.death_context_patterns:
            for match in pattern.findall(sentence):
                value = match if isinstance(match, str) else match[0]
                normalized = normalize_text(value)
                if normalized:
                    values.append(normalized)
        if values:
            return list(dict.fromkeys(values))
        if re.search(rf"\b{re.escape(entity_text)}\b[^.!?]{{0,80}}\bdied on\b", sentence, re.IGNORECASE):
            return self.full_dates(sentence)
        return []

    def person_aliases(self, sentence: str, entity_text: str) -> List[str]:
        values: List[str] = []
        entity_norm = normalize_text(entity_text)

        for match in self.parenthetical_alias_pattern.findall(sentence):
            base_name = normalize_text(match[0])
            alias_candidate = normalize_text(match[1])
            if base_name.lower() != entity_norm.lower():
                continue
            if self.is_valid_person_alias(alias_candidate, entity_norm):
                values.append(alias_candidate)

        for match in self.quoted_alias_pattern.findall(sentence):
            alias_candidate = normalize_text(match)
            if self.is_valid_person_alias(alias_candidate, entity_norm):
                values.append(alias_candidate)

        for match in self.alias_pattern.findall(sentence):
            alias_candidate = normalize_text(match)
            if self.is_valid_person_alias(alias_candidate, entity_norm):
                values.append(alias_candidate)

        return list(dict.fromkeys(values))

    def is_valid_person_alias(self, alias_text: str, entity_text: str) -> bool:
        alias_norm = normalize_text(alias_text)
        entity_norm = normalize_text(entity_text)
        if not alias_norm:
            return False
        if alias_norm.lower() == entity_norm.lower():
            return False
        if len(alias_norm) < 3 or len(alias_norm) > 40:
            return False
        if re.search(r"\b(?:theory|machine|degree|paper|test|concept|work|design)\b", alias_norm, re.IGNORECASE):
            return False
        if re.search(r"\d{4}", alias_norm):
            return False
        if alias_norm.lower().startswith("the "):
            return False
        if not re.fullmatch(r"[A-Za-z][A-Za-z .'\-]{1,39}", alias_norm):
            return False
        return True

    def person_occupations(self, sentence: str, entity_text: str) -> List[str]:
        if not self.sentence_mentions_person(sentence, entity_text):
            return []
        match = self.lead_occupation_pattern.search(sentence)
        if not match:
            return []

        phrase = normalize_text(match.group(1)).lower()
        values: List[str] = []
        for occupation in OCCUPATIONS:
            if re.search(rf"\b{re.escape(occupation)}\b", phrase, re.IGNORECASE):
                values.append(occupation)

        return list(dict.fromkeys(values))

    def rule_attribute_confidence(self, label: str, attr_name: str, sentence: str, entity_text: str, attr_value: str) -> Optional[float]:
        sentence_norm = normalize_text(sentence)
        if label == "Person":
            if attr_name == "birth_date" and self.sentence_mentions_person(sentence_norm, entity_text) and re.search(r"\b(?:born|birth)\b", sentence_norm, re.IGNORECASE):
                return 0.97
            if attr_name == "death_date" and self.sentence_mentions_person(sentence_norm, entity_text) and re.search(r"\b(?:died|death|passed away)\b", sentence_norm, re.IGNORECASE):
                return 0.97
            if attr_name == "occupation" and self.sentence_mentions_person(sentence_norm, entity_text) and re.search(r"\b(?:is|was)\s+(?:an?|the)\b", sentence_norm, re.IGNORECASE):
                return 0.94
        if label == "Organization" and attr_name == "org_type":
            return 0.93
        return None

    def full_dates(self, sentence: str) -> List[str]:
        return list(dict.fromkeys(self.full_date_pattern.findall(sentence)))

    def years(self, sentence: str) -> List[str]:
        return list(dict.fromkeys(self.year_pattern.findall(sentence)))

    def aliases(self, sentence: str) -> List[str]:
        return [normalize_text(value) for value in self.alias_pattern.findall(sentence) if normalize_text(value)]

    def locations(self, sentence: str) -> List[str]:
        values = []
        for match in self.location_pattern.findall(sentence):
            value = normalize_text(match)
            if value:
                values.append(value)
        return list(dict.fromkeys(values))

    def venues(self, sentence: str) -> List[str]:
        values = []
        for match in self.venue_pattern.findall(sentence):
            value = normalize_text(match)
            if value and any(hint in value.lower() for hint in VENUE_HINTS):
                values.append(value)
        return list(dict.fromkeys(values))

    def description(self, sentence: str) -> Optional[str]:
        match = self.description_pattern.search(sentence)
        if not match:
            return None
        value = normalize_text(match.group(1))
        return value if len(value) >= 5 else None


class TypedNLIAttributeExtractor:
    def __init__(self, model_name: str = "cross-encoder/nli-distilroberta-base", device: str = "cpu"):
        print(f"Loading NLI Model for Attribute Extraction: {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.device = device

    def build_hypothesis(self, ent_name: str, ent_label: str, attr_name: str, attr_value: str) -> Optional[str]:
        if attr_name == "birth_date" and ent_label == "Person":
            return f"{ent_name} was born on {attr_value}."
        if attr_name == "death_date" and ent_label == "Person":
            return f"{ent_name} died on {attr_value}."
        if attr_name == "nationality" and ent_label == "Person":
            return f"{ent_name} is {attr_value}."
        if attr_name == "occupation" and ent_label == "Person":
            return f"{ent_name} was a {attr_value}."
        if attr_name == "aliases":
            return f"{ent_name} is also known as {attr_value}."
        if attr_name == "established_year" and ent_label == "Organization":
            return f"{ent_name} was established in {attr_value}."
        if attr_name == "org_type" and ent_label == "Organization":
            return f"{ent_name} is a {attr_value}."
        if attr_name == "location" and ent_label == "Organization":
            return f"{ent_name} is located in {attr_value}."
        if attr_name == "publish_year" and ent_label == "Publication":
            return f"{ent_name} was published in {attr_value}."
        if attr_name == "venue" and ent_label == "Publication":
            return f"{ent_name} was published in {attr_value}."
        if attr_name == "start_date" and ent_label == "Event":
            return f"{ent_name} started in {attr_value}."
        if attr_name == "end_date" and ent_label == "Event":
            return f"{ent_name} ended in {attr_value}."
        if attr_name == "creation_year" and ent_label in {"Artifact", "Concept"}:
            return f"{ent_name} was created in {attr_value}."
        if attr_name == "description" and ent_label in {"Artifact", "Concept"}:
            return f"{ent_name} is {attr_value}."
        return None

    def threshold_for_attribute(self, attr_name: str, default_threshold: float) -> float:
        return ATTRIBUTE_NLI_THRESHOLDS.get(attr_name, default_threshold)

    def verify_attributes_batch(self, sentence: str, entity: Dict[str, str], candidates: Dict[str, List[str]], threshold: float) -> List[Dict]:
        ent_name = canonicalize_entity_text(entity["text"], entity["label"])
        ent_label = entity["label"]
        rule_helper = TypedCandidateExtractor()
        hypotheses_data: List[Tuple[str, str, str, str]] = []
        extracted: List[Dict] = []

        for attr_name, values in candidates.items():
            if attr_name not in ALLOWED_ATTRIBUTES.get(ent_label, []):
                continue
            for value in values:
                rule_conf = rule_helper.rule_attribute_confidence(ent_label, attr_name, sentence, ent_name, str(value))
                if rule_conf is not None:
                    extracted.append(
                        {
                            "entity_label": ent_label,
                            "entity_text": ent_name,
                            "attribute_name": attr_name,
                            "attribute_value": value,
                            "confidence": round(rule_conf, 4),
                        }
                    )
                    continue
                hypothesis = self.build_hypothesis(ent_name, ent_label, attr_name, value)
                if hypothesis:
                    hypotheses_data.append((sentence, hypothesis, attr_name, value))

        if not hypotheses_data:
            return extracted

        inputs = self.tokenizer(
            [item[0] for item in hypotheses_data],
            [item[1] for item in hypotheses_data],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

        best_singletons: Dict[str, Tuple[str, float]] = {}

        for idx, (_, _, attr_name, attr_value) in enumerate(hypotheses_data):
            entailment_prob = probs[idx][2].item()
            if entailment_prob < self.threshold_for_attribute(attr_name, threshold):
                continue

            if attr_name in {"birth_date", "death_date", "established_year", "publish_year", "start_date", "end_date", "creation_year", "org_type", "location", "venue", "description"}:
                if attr_name not in best_singletons or entailment_prob > best_singletons[attr_name][1]:
                    best_singletons[attr_name] = (attr_value, entailment_prob)
            else:
                extracted.append(
                    {
                        "entity_label": ent_label,
                        "entity_text": ent_name,
                        "attribute_name": attr_name,
                        "attribute_value": attr_value,
                        "confidence": round(entailment_prob, 4),
                    }
                )

        for attr_name, (attr_value, prob) in best_singletons.items():
            final_value = int(attr_value) if attr_name in {"established_year", "publish_year", "creation_year"} and str(attr_value).isdigit() else attr_value
            extracted.append(
                {
                    "entity_label": ent_label,
                    "entity_text": ent_name,
                    "attribute_name": attr_name,
                    "attribute_value": final_value,
                    "confidence": round(prob, 4),
                }
            )

        return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/turing_schema_corpus.jsonl")
    parser.add_argument("--ner-dir", default="data/output/ner_bilstm_crf")
    parser.add_argument("--output", default="data/output/attributes.jsonl")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--nli-model", default="cross-encoder/nli-distilroberta-base")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        raise SystemExit("Attribute extraction requires 'torch' to be installed.")
    if not TRANSFORMERS_AVAILABLE:
        raise SystemExit("Attribute extraction requires 'transformers' to be installed.")

    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner_model, cv, tv = ner_module.load_ner_model_standalone(Path(args.ner_dir), model_device)
    extractor = TypedNLIAttributeExtractor(model_name=args.nli_model, device=model_device)
    candidate_extractor = TypedCandidateExtractor()

    with open(args.input, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    all_attributes: List[Dict] = []
    seen_attr_keys = set()

    for row in tqdm(rows, desc="Extracting Attributes (Typed NLI)"):
        text = row.get("text", "")
        doc_id = str(row.get("doc_id", row.get("id", "")))
        source = str(row.get("url", "")) or doc_id
        page_subject = infer_page_subject_entity(doc_id)

        for sent_idx, sent_text in enumerate(split_sentences(text)):
            entities = ner_module.predict_entities(ner_model, cv, tv, sent_text, model_device)
            if page_subject:
                entities = [page_subject] + entities
            if not entities:
                continue

            seen_entities = set()

            for entity in entities:
                if not is_valid_attribute_entity(entity):
                    continue
                entity["text"] = canonicalize_entity_text(entity["text"], entity["label"])
                entity_key = (entity["label"], entity["text"])
                if entity_key in seen_entities:
                    continue
                seen_entities.add(entity_key)
                candidates = candidate_extractor.extract(sent_text, entity)
                if not candidates:
                    continue

                valid_attrs = extractor.verify_attributes_batch(sent_text, entity, candidates, args.threshold)
                for attr in valid_attrs:
                    attr_key = (
                        attr["entity_label"],
                        attr["entity_text"],
                        attr["attribute_name"],
                        str(attr["attribute_value"]),
                        doc_id,
                        sent_idx,
                    )
                    if attr_key in seen_attr_keys:
                        continue
                    seen_attr_keys.add(attr_key)
                    attr["entity_uid"] = f"{attr['entity_label']}:{slugify(attr['entity_text'])}"
                    attr["doc_id"] = doc_id
                    attr["sent_id"] = f"{doc_id}__sent{sent_idx:03d}"
                    attr["evidence"] = sent_text
                    attr["source"] = source
                    attr["extract_method"] = "typed_candidate_plus_nli"
                    all_attributes.append(attr)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f_out:
        for attr in all_attributes:
            f_out.write(json.dumps(attr, ensure_ascii=False) + "\n")

    print(f"[DONE] Extracted {len(all_attributes)} attributes.")
    print(f"[DONE] Saved to {args.output}")


if __name__ == "__main__":
    main()
