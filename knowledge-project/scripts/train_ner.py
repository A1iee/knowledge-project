from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    AdamW = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from tqdm import tqdm

try:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModelForTokenClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    TRANSFORMERS_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CORPUS_PATH = REPO_ROOT / "data" / "raw" / "turing_schema_corpus.jsonl"
INTERMEDIATE_PATH = REPO_ROOT / "data" / "intermediate" / "ner_char_bio.jsonl"
MODEL_DIR = REPO_ROOT / "data" / "output" / "ner"
DEFAULT_BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256

ENTITY_LABELS = [
    "Person",
    "Organization",
    "Location",
    "Concept",
    "Artifact",
    "Event",
    "Publication",
    "Honor",
]

TAG_LIST = ["O"] + [f"{prefix}-{label}" for label in ENTITY_LABELS for prefix in ("B", "I")]
CENTER_ALIASES = [
    ("Alan Mathison Turing", "Person"),
    ("Alan Turing", "Person"),
    ("A. M. Turing", "Person"),
    ("Turing", "Person"),
]

DEFAULT_GAZETTEER: Dict[str, List[str]] = {
    "Person": [
        "Alan Turing",
        "Alan Mathison Turing",
        "A. M. Turing",
        "Joan Clarke",
        "Alonzo Church",
        "John von Neumann",
        "Claude Shannon",
        "Max Newman",
        "Donald Michie",
        "I. J. Good",
        "Gordon Welchman",
        "Hugh Alexander",
        "Robin Gandy",
    ],
    "Organization": [
        "King's College, Cambridge",
        "University of Cambridge",
        "Princeton University",
        "Bletchley Park",
        "University of Manchester",
        "Government Code and Cypher School",
        "National Physical Laboratory",
        "Hut 8",
        "Bell Labs",
        "Foreign Office",
        "Computing Machine Laboratory",
    ],
    "Location": [
        "London",
        "Manchester",
        "Princeton",
        "Princeton, New Jersey",
        "Maida Vale",
        "Wilmslow",
        "Bletchley",
        "Cambridge",
        "Oxford",
        "England",
        "Britain",
        "United Kingdom",
    ],
    "Concept": [
        "Turing machine",
        "Turing test",
        "Halting problem",
        "Church-Turing thesis",
        "Computability theory",
        "Morphogenesis",
        "Universal Turing machine",
        "Decision problem",
        "Oracle machine",
        "Imitation game",
        "Banburismus",
    ],
    "Artifact": [
        "Bombe",
        "bombe",
        "Automatic Computing Engine",
        "Enigma machine",
        "Manchester Baby",
        "Colossus computer",
        "Pilot ACE",
        "Ultra intelligence",
    ],
    "Event": [
        "World War II",
        "Battle of the Atlantic",
        "Royal pardon",
        "Alan Turing law",
        "Turing centenary",
        "Second World War",
        "1952 Conviction",
    ],
    "Publication": [
        "On Computable Numbers",
        "Computing Machinery and Intelligence",
        "Systems of Logic Based on Ordinals",
        "The Chemical Basis of Morphogenesis",
        "Intelligent Machinery",
        "Can Digital Computers Think?",
        "The Applications of Probability to Cryptography",
        "the chemical basis of morphogenesis",
        "Computable Numbers",
        "First Draft of a Report on the EDVAC",
        "Proposed Electronic Calculator",
        "\"Computing Machinery and Intelligence\"",
    ],
    "Honor": [
        "Turing Award",
        "Member of the Order of the British Empire",
        "Order of the British Empire",
        "MBE",
        "OBE",
        "Officer of the Order of the British Empire",
        "Member of the British Empire",
        "British Empire award",
        "award",
        "prize",
        "medal",
        "honour",
        "honor",
        "appointment",
        "royal pardon",
        "posthumous pardon",
        "posthumous royal pardon",
        "commemorative blue plaque",
        "blue plaque",
        "memorial",
        "statue",
        "Bank of England note",
        "Bank of England PS50 note",
        "Bank of England 50 pound note",
        "centenary celebrations",
        "Bank of England £50 note",
        "Bank of England 50 note",
        "Alan Turing law",
        "BBC series",
    ],
}

HONOR_CAPTURE_PATTERNS = [
    r"\bappointed as (?:a |an )?(Member of the Order of the British Empire|Officer of the Order of the British Empire)\b",
    r"\bappointment as (?:a |an )?(Member of the Order of the British Empire|Officer of the Order of the British Empire)\b",
    r"\bsuch as appointment as (?:a |an )?(Member of the Order of the British Empire|Officer of the Order of the British Empire)\b",
    r"\bawarded the ([A-Z][A-Za-z'’\- ]{2,100}(?:Award|Prize|Medal|Honour|Honor))\b",
    r"\breceived the ([A-Z][A-Za-z'’\- ]{2,100}(?:Award|Prize|Medal|Honour|Honor|Pardon))\b",
    r"\bwon the ([A-Z][A-Za-z'’\- ]{2,100}(?:Award|Prize|Medal))\b",
    r"\bhonoured with (?:the )?([A-Z][A-Za-z'’\- ]{2,100}(?:Award|Prize|Medal|Honour|Honor))\b",
    r"\bcommemorated by (?:the )?([A-Z][A-Za-z'’\- ]{2,100})\b",
    r"\bfeatured on (?:the )?(Bank of England [A-Za-z0-9£$ ]{2,40}note)\b",
    r"\breceived a (posthumous [A-Za-z'’\- ]{2,60}pardon)\b",
    r"\bwas honoured on (?:the )?([A-Z][A-Za-z0-9£$'’\- ]{2,100})\b",
]

PUBLICATION_CAPTURE_PATTERNS = [
    r'\bin\s+["“]([^"”]{3,120})["”]',
    r'\bpaper\s+["“]([^"”]{3,120})["”]',
    r'\bessay\s+["“]([^"”]{3,120})["”]',
    r"\bwrote on ([A-Za-z][A-Za-z'’\- ]{3,120})",
    r"\bpublished a paper on ([A-Za-z][A-Za-z'’\- ]{3,120})",
]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def device():
    if not TORCH_AVAILABLE:
        return "cpu"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_jsonl(path: str | Path) -> Iterable[Dict]:
    with resolve(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(rows: Iterable[Dict], path: str | Path) -> None:
    target = resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def safe_label(label: str) -> Optional[str]:
    return label if label in ENTITY_LABELS else None


def build_tag_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    tag_to_id = {tag: idx for idx, tag in enumerate(TAG_LIST)}
    id_to_tag = {idx: tag for tag, idx in tag_to_id.items()}
    return tag_to_id, id_to_tag


def load_gazetteer() -> Dict[str, List[str]]:
    gazetteer = {label: list(values) for label, values in DEFAULT_GAZETTEER.items()}
    for alias, label in CENTER_ALIASES:
        gazetteer.setdefault(label, [])
        if alias not in gazetteer[label]:
            gazetteer[label].append(alias)
    return gazetteer


def find_non_overlapping_spans(text: str, phrase: str) -> List[Tuple[int, int]]:
    if not phrase:
        return []
    pattern = re.compile(rf"(?<![A-Za-z]){re.escape(phrase)}(?![A-Za-z])", re.IGNORECASE)
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def apply_bio_span(tags: List[str], start: int, end: int, label: str) -> None:
    if start >= end or start < 0 or end > len(tags):
        return
    if any(tag != "O" for tag in tags[start:end]):
        return
    tags[start] = f"B-{label}"
    for idx in range(start + 1, end):
        tags[idx] = f"I-{label}"


def gazetteer_matches(text: str, gazetteer: Dict[str, List[str]]) -> List[Tuple[int, int, str]]:
    matches: List[Tuple[int, int, str]] = []
    for label, phrases in gazetteer.items():
        for phrase in phrases:
            for start, end in find_non_overlapping_spans(text, phrase):
                matches.append((start, end, label))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    return matches


def rule_pattern_matches(text: str) -> List[Tuple[int, int, str]]:
    matches: List[Tuple[int, int, str]] = []

    person_name = r"[A-Z][A-Za-z.\-']*(?:\s+[A-Z][A-Za-z.\-']*){0,4}"
    org_name = r"[A-Z][A-Za-z&.\-']*(?:\s+[A-Z][A-Za-z&.\-']*){0,6}"
    location_name = r"[A-Z][A-Za-z.\-']*(?:\s+[A-Z][A-Za-z.\-']*){0,4}"

    person_loc_patterns = [
        (rf"({person_name}) was born in ({location_name})", "Person", "Location"),
        (rf"({person_name}) graduated from ({org_name})", "Person", "Organization"),
        (rf"({person_name}) studied at ({org_name})", "Person", "Organization"),
        (rf"({person_name}) worked at ({org_name})", "Person", "Organization"),
        (rf"({person_name}) worked for ({org_name})", "Person", "Organization"),
    ]
    for pattern, head_label, tail_label in person_loc_patterns:
        for match in re.finditer(pattern, text):
            head_span = match.span(1)
            tail_span = match.span(2)
            matches.append((head_span[0], head_span[1], head_label))
            matches.append((tail_span[0], tail_span[1], tail_label))

    for match in re.finditer(r"\(([^()]{2,80})\)", text):
        candidate = normalize_space(match.group(1))
        if re.fullmatch(r"[A-Z][A-Za-z.\- ]{1,60}", candidate):
            matches.append((match.start(1), match.end(1), "Person"))

    for pattern in HONOR_CAPTURE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.span(1)
            matches.append((start, end, "Honor"))

    for pattern in PUBLICATION_CAPTURE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            candidate = normalize_space(match.group(1)).strip(".,;: ")
            if len(candidate) < 4:
                continue
            if candidate.lower().startswith(("the problem of ", "the possibility of ")):
                continue
            start, end = match.span(1)
            matches.append((start, end, "Publication"))

    return matches

def weak_label_text(text: str, seed_label: Optional[str], seed_title: Optional[str], title: Optional[str], gazetteer: Dict[str, List[str]]) -> List[str]:
    tags = ["O"] * len(text)

    local_gazetteer = {label: list(values) for label, values in gazetteer.items()}
    for candidate, label in [(seed_title, seed_label), (title, seed_label)]:
        if candidate and safe_label(label or ""):
            local_gazetteer.setdefault(label, [])
            if candidate not in local_gazetteer[label]:
                local_gazetteer[label].append(candidate)

    matches = gazetteer_matches(text, local_gazetteer)
    matches.extend(rule_pattern_matches(text))
    matches.sort(key=lambda item: (-(item[1] - item[0]), item[0]))

    for start, end, label in matches:
        if not safe_label(label):
            continue
        apply_bio_span(tags, start, end, label)

    return tags


def prepare_dataset(args: argparse.Namespace) -> None:
    gazetteer = load_gazetteer()
    rows: List[Dict] = []

    for idx, row in enumerate(read_jsonl(args.input)):
        text = row.get("text", "")
        if not text:
            continue
        sample_id = str(row.get("doc_id", row.get("id", f"sample_{idx:06d}")))
        tags = weak_label_text(
            text=text,
            seed_label=row.get("seed_label"),
            seed_title=row.get("seed_title"),
            title=row.get("title"),
            gazetteer=gazetteer,
        )
        rows.append({"id": f"{sample_id}__seg000", "text": text, "tags": tags})

    write_jsonl(rows, args.output)
    print(f"[DONE] Prepared {len(rows)} weakly labeled NER samples -> {resolve(args.output)}")


def char_tags_to_token_tags(offsets: Sequence[Tuple[int, int]], char_tags: Sequence[str]) -> List[str]:
    token_tags: List[str] = []
    prev_entity: Optional[str] = None

    for start, end in offsets:
        if start == end:
            token_tags.append("O")
            continue

        segment = [tag for tag in char_tags[start:end] if tag != "O"]
        if not segment:
            token_tags.append("O")
            prev_entity = None
            continue

        entity_names = [tag.split("-", 1)[1] for tag in segment if "-" in tag]
        if not entity_names:
            token_tags.append("O")
            prev_entity = None
            continue

        entity = max(set(entity_names), key=entity_names.count)
        prefix = "B"
        first_char_tag = next((tag for tag in char_tags[start:end] if tag != "O"), "O")
        if first_char_tag.startswith("I-") and prev_entity == entity:
            prefix = "I"
        elif prev_entity == entity and first_char_tag.startswith(("B-", "I-")):
            prefix = "I"

        token_tag = f"{prefix}-{entity}"
        token_tags.append(token_tag)
        prev_entity = entity

    return token_tags


@dataclass
class TokenizedSample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


class TokenNERDataset(Dataset):
    def __init__(self, samples: List[TokenizedSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch is required for dataset access.")
        sample = self.samples[index]
        return {
            "input_ids": torch.tensor(sample.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(sample.attention_mask, dtype=torch.long),
            "labels": torch.tensor(sample.labels, dtype=torch.long),
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for batching.")
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def tokenize_training_samples(samples: List[Dict], tokenizer, tag_to_id: Dict[str, int], max_length: int) -> List[TokenizedSample]:
    tokenized_samples: List[TokenizedSample] = []

    for sample in samples:
        text = sample["text"]
        char_tags = sample["tags"]
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        offsets = encoded.pop("offset_mapping")
        token_tags = char_tags_to_token_tags(offsets, char_tags)
        labels = [tag_to_id.get(tag, tag_to_id["O"]) if offset != (0, 0) else -100 for tag, offset in zip(token_tags, offsets)]
        tokenized_samples.append(
            TokenizedSample(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=labels,
            )
        )

    return tokenized_samples


def decode_valid_tag_ids(label_ids: Sequence[int], id_to_tag: Dict[int, str]) -> List[str]:
    return [id_to_tag.get(int(label_id), "O") for label_id in label_ids if label_id != -100]


def bio_tags_to_spans(tags: Sequence[str]) -> set[Tuple[str, int, int]]:
    spans: set[Tuple[str, int, int]] = set()
    start_idx: Optional[int] = None
    current_label: Optional[str] = None

    def close_span(end_idx: int) -> None:
        nonlocal start_idx, current_label
        if start_idx is not None and current_label is not None:
            spans.add((current_label, start_idx, end_idx))
        start_idx = None
        current_label = None

    for idx, tag in enumerate(tags):
        if tag == "O" or "-" not in tag:
            close_span(idx)
            continue

        prefix, label = tag.split("-", 1)
        if prefix == "B":
            close_span(idx)
            start_idx = idx
            current_label = label
            continue

        if prefix == "I" and start_idx is not None and current_label == label:
            continue

        close_span(idx)
        if prefix == "I":
            start_idx = idx
            current_label = label

    close_span(len(tags))
    return spans


def compute_dev_prf1(
    gold_sequences: Sequence[Sequence[int]],
    pred_sequences: Sequence[Sequence[int]],
    id_to_tag: Dict[int, str],
) -> Dict[str, float]:
    gold_total = 0
    pred_total = 0
    true_positive = 0

    for gold_ids, pred_ids in zip(gold_sequences, pred_sequences):
        gold_tags = decode_valid_tag_ids(gold_ids, id_to_tag)
        pred_tags = decode_valid_tag_ids(pred_ids, id_to_tag)
        seq_len = min(len(gold_tags), len(pred_tags))
        gold_spans = bio_tags_to_spans(gold_tags[:seq_len])
        pred_spans = bio_tags_to_spans(pred_tags[:seq_len])
        gold_total += len(gold_spans)
        pred_total += len(pred_spans)
        true_positive += len(gold_spans & pred_spans)

    precision = true_positive / pred_total if pred_total else 0.0
    recall = true_positive / gold_total if gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gold_entities": float(gold_total),
        "pred_entities": float(pred_total),
        "true_positive": float(true_positive),
    }


def save_metadata(output_dir: Path, tag_to_id: Dict[str, int], gazetteer: Dict[str, List[str]], base_model: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "tag_vocab.json").open("w", encoding="utf-8") as f:
        json.dump({"itos": TAG_LIST}, f, ensure_ascii=False, indent=2)
    with (output_dir / "char_vocab.json").open("w", encoding="utf-8") as f:
        json.dump({"itos": ["<tokenizer-based>"]}, f, ensure_ascii=False, indent=2)
    with (output_dir / "gazetteer.json").open("w", encoding="utf-8") as f:
        json.dump(gazetteer, f, ensure_ascii=False, indent=2)
    with (output_dir / "ner_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": base_model,
                "max_length": MAX_LENGTH,
                "labels": ENTITY_LABELS,
                "center_aliases": [alias for alias, _ in CENTER_ALIASES],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def train_model(args: argparse.Namespace) -> None:
    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        raise SystemExit("Training requires both 'torch' and 'transformers' to be installed.")

    training_path = resolve(args.data)
    if not training_path.exists():
        raise SystemExit(f"Training data not found: {training_path}")

    from sklearn.model_selection import train_test_split

    samples = list(read_jsonl(training_path))
    if not samples:
        raise SystemExit("Training data is empty.")
    tag_to_id, id_to_tag = build_tag_mappings()
    model_device = device()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(TAG_LIST),
        id2label=id_to_tag,
        label2id=tag_to_id,
    ).to(model_device)
    train_samples, dev_samples = train_test_split(samples, test_size=0.15, random_state=42)
    train_dataset = TokenNERDataset(tokenize_training_samples(train_samples, tokenizer, tag_to_id, args.max_length))
    dev_dataset = TokenNERDataset(tokenize_training_samples(dev_samples, tokenizer, tag_to_id, args.max_length))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"NER Train Epoch {epoch + 1}/{args.epochs}"):
            batch = {k: v.to(model_device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_loader), 1)
        model.eval()
        all_pred_sequences: List[List[int]] = []
        all_gold_sequences: List[List[int]] = []
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(model_device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                for p, l in zip(preds, labels):
                    valid_pred = [int(pi) for pi, li in zip(p, l) if li != -100]
                    valid_gold = [int(li) for li in l if li != -100]
                    all_pred_sequences.append(valid_pred)
                    all_gold_sequences.append(valid_gold)
        metrics = compute_dev_prf1(all_gold_sequences, all_pred_sequences, id_to_tag)
        print(
            f"[INFO] Epoch {epoch+1} avg loss: {avg_loss:.4f} | "
            f"Dev P: {metrics['precision']:.4f} "
            f"R: {metrics['recall']:.4f} "
            f"F1: {metrics['f1']:.4f} | "
            f"Gold: {int(metrics['gold_entities'])} "
            f"Pred: {int(metrics['pred_entities'])} "
            f"TP: {int(metrics['true_positive'])}"
        )
        model.train()
    output_dir = resolve(args.output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_metadata(output_dir, tag_to_id, load_gazetteer(), args.base_model)
    print(f"[DONE] Saved NER model to {output_dir}")


class HybridNERModel:
    def __init__(
        self,
        model,
        tokenizer,
        tag_to_id: Dict[str, int],
        id_to_tag: Dict[int, str],
        gazetteer: Dict[str, List[str]],
        model_device: Any,
        max_length: int = MAX_LENGTH,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tag_to_id = tag_to_id
        self.id_to_tag = id_to_tag
        self.gazetteer = gazetteer
        self.device = model_device
        self.max_length = max_length


def load_tag_vocab(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        tags = data.get("itos", TAG_LIST)
    else:
        tags = TAG_LIST
    tag_to_id = {tag: idx for idx, tag in enumerate(tags)}
    id_to_tag = {idx: tag for tag, idx in tag_to_id.items()}
    return tag_to_id, id_to_tag


def load_ner_model_standalone(model_dir: Path, model_device):
    model_dir = resolve(model_dir)
    tag_to_id, id_to_tag = load_tag_vocab(model_dir / "tag_vocab.json")
    gazetteer_path = model_dir / "gazetteer.json"
    gazetteer = load_gazetteer()
    if gazetteer_path.exists():
        with gazetteer_path.open("r", encoding="utf-8") as f:
            gazetteer = json.load(f)

    model = None
    tokenizer = None
    if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE and (model_dir / "config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir).to(model_device)
        model.eval()

    hybrid = HybridNERModel(
        model=model,
        tokenizer=tokenizer,
        tag_to_id=tag_to_id,
        id_to_tag=id_to_tag,
        gazetteer=gazetteer,
        model_device=model_device,
    )
    return hybrid, None, None


def tags_to_entities(text: str, offsets: Sequence[Tuple[int, int]], tag_ids: Sequence[int], id_to_tag: Dict[int, str]) -> List[Dict]:
    entities: List[Dict] = []
    current: Optional[Dict] = None

    for offset, tag_id in zip(offsets, tag_ids):
        if offset == (0, 0) or tag_id < 0:
            continue

        tag = id_to_tag.get(int(tag_id), "O")
        if tag == "O":
            if current:
                entities.append(current)
                current = None
            continue

        prefix, label = tag.split("-", 1)
        start, end = offset

        if prefix == "B" or current is None or current["label"] != label or start > current["end"] + 1:
            if current:
                entities.append(current)
            current = {"text": text[start:end], "label": label, "start": start, "end": end}
        else:
            current["end"] = end
            current["text"] = text[current["start"]:current["end"]]

    if current:
        entities.append(current)
    return entities


def merge_entities(entities: List[Dict]) -> List[Dict]:
    by_key: Dict[Tuple[int, int, str], Dict] = {}
    for entity in entities:
        key = (entity["start"], entity["end"], entity["label"])
        by_key[key] = entity

    deduped = list(by_key.values())
    deduped.sort(key=lambda item: (-(item["end"] - item["start"]), item["start"], item["label"]))

    selected: List[Dict] = []
    for candidate in deduped:
        overlaps = False
        for existing in selected:
            same_span = candidate["start"] == existing["start"] and candidate["end"] == existing["end"]
            nested = candidate["start"] >= existing["start"] and candidate["end"] <= existing["end"]
            if same_span and candidate["label"] == existing["label"]:
                overlaps = True
                break
            if nested:
                overlaps = True
                break
        if not overlaps:
            selected.append(candidate)

    selected.sort(key=lambda item: (item["start"], item["end"]))
    return selected


def clean_rule_entity(entity: Dict) -> Optional[Dict]:
    text = normalize_space(str(entity.get("text", ""))).strip(".,;:()[]{}\"'")
    label = entity.get("label", "")
    if not text:
        return None
    if label == "Honor" and text.lower() in {"award", "prize", "medal", "honour", "honor", "appointment"}:
        return None
    if label == "Publication" and len(text) < 6:
        return None
    cleaned = dict(entity)
    cleaned["text"] = text
    return cleaned


def extract_rule_entities(text: str, gazetteer: Dict[str, List[str]]) -> List[Dict]:
    entities: List[Dict] = []
    for label, phrases in gazetteer.items():
        for phrase in phrases:
            for start, end in find_non_overlapping_spans(text, phrase):
                entities.append({"text": text[start:end], "label": label, "start": start, "end": end})

    for alias, label in CENTER_ALIASES:
        for start, end in find_non_overlapping_spans(text, alias):
            entities.append({"text": text[start:end], "label": label, "start": start, "end": end})

    for start, end, label in rule_pattern_matches(text):
        entities.append({"text": text[start:end], "label": label, "start": start, "end": end})

    cleaned_entities = []
    for entity in entities:
        cleaned = clean_rule_entity(entity)
        if cleaned:
            cleaned_entities.append(cleaned)

    return merge_entities(cleaned_entities)


def predict_entities(model_wrapper: HybridNERModel, _cv, _tv, text: str, model_device) -> List[Dict]:
    rule_entities = extract_rule_entities(text, model_wrapper.gazetteer)

    if not TORCH_AVAILABLE or model_wrapper.model is None or model_wrapper.tokenizer is None:
        return rule_entities

    with torch.no_grad():
        encoded = model_wrapper.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            padding="max_length",
            max_length=model_wrapper.max_length,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        inputs = {key: value.to(model_device) for key, value in encoded.items()}
        logits = model_wrapper.model(**inputs).logits[0]
        pred_ids = torch.argmax(logits, dim=-1).tolist()

    model_entities = tags_to_entities(text, offsets, pred_ids, model_wrapper.id_to_tag)
    return merge_entities(model_entities + rule_entities)


def demo_predict(args: argparse.Namespace) -> None:
    model_wrapper, cv, tv = load_ner_model_standalone(resolve(args.model_dir), device())
    entities = predict_entities(model_wrapper, cv, tv, args.text, device())
    print(json.dumps(entities, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "train", "predict"])
    parser.add_argument("--input", default=str(RAW_CORPUS_PATH.relative_to(REPO_ROOT)))
    parser.add_argument("--data", default=str(INTERMEDIATE_PATH.relative_to(REPO_ROOT)))
    parser.add_argument("--output", default=str(INTERMEDIATE_PATH.relative_to(REPO_ROOT)))
    parser.add_argument("--output-dir", default=str(MODEL_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--model-dir", default=str(MODEL_DIR.relative_to(REPO_ROOT)))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--text", default="Alan Turing graduated from King's College, Cambridge and worked at Bletchley Park.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare_dataset(args)
    elif args.mode == "train":
        train_model(args)
    elif args.mode == "predict":
        demo_predict(args)


if __name__ == "__main__":
    main()
