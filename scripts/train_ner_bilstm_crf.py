"""BiLSTM-CRF NER training/inference for turing_schema_corpus.jsonl."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Constants & Schema Seeds
# ---------------------------------------------------------------------------
# 弱监督的种子实体词典，用于在未标注文本中通过精准匹配生成训练标签(BIO格式)
SCHEMA_SEEDS: Dict[str, List[str]] = {
    "Person": ["Alan Turing", "Joan Clarke", "Alonzo Church", "John von Neumann", "Claude Shannon", "Max Newman", "Donald Michie", "I. J. Good", "Gordon Welchman", "Hugh Alexander", "Irving John Good", "Robin Gandy"],
    "Organization": ["King's College, Cambridge", "University of Cambridge", "Princeton University", "Bletchley Park", "University of Manchester", "Government Code and Cypher School", "National Physical Laboratory (United Kingdom)", "Hut 8", "Bell Labs", "Foreign Office", "Balliol College, Oxford", "Balliol College", "University of Oxford", "Rugby School", "Abbotsholme School", "Budapest University", "Heidelberg University", "National Defense Research Committee", "National Defense Research Committee (NDRC)"],
    "Location": ["London", "Manchester", "Princeton, New Jersey", "Maida Vale", "Wilmslow", "Bletchley", "Cambridge", "Sherborne School", "Oxford", "Oxfordshire", "Derbyshire", "Budapest", "Rangoon", "Burma", "Hungary", "Gaylord, Michigan", "Petoskey", "Michigan"],
    "Concept": ["Turing machine", "Turing test", "Halting problem", "Church-Turing thesis", "Computability theory", "Morphogenesis", "Universal Turing machine", "Decision problem", "Oracle machine", "Imitation game", "Turing reduction"],
    "Artifact": ["Bombe", "Automatic Computing Engine", "Enigma machine", "Manchester Baby", "Colossus computer", "Pilot ACE", "Universal machine"],
    "Event": ["World War II", "Cryptanalysis of the Enigma", "Alan Turing law", "Royal pardon", "Turing centenary", "Prosecution of Alan Turing", "Second Boer War"],
    "Publication": ["On Computable Numbers", "Computing Machinery and Intelligence", "Systems of Logic Based on Ordinals", "The Chemical Basis of Morphogenesis", "Intelligent Machinery", "Can Digital Computers Think?", "The Applications of Probability to Cryptography"],
    "Honor": ["Turing Award", "Order of the British Empire", "Bank of England 50 note", "Alan Turing Year", "Turing's law"],
}

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def resolve_path(p: str) -> Path:
    """解析路径：若是相对路径，则基于项目根目录(REPO_ROOT)进行拼接"""
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()

def get_device() -> torch.device:
    """自动推断最优硬件：CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU"""
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _safe_ckpt_config(args: argparse.Namespace) -> dict:
    """过滤掉 argparse 中的函数对象，防止模型保存(pickle)时报错"""
    cfg = dict(vars(args))
    cfg.pop("func", None)
    return cfg

def _torch_load_compat(path: Path, device: torch.device):
    """兼容 PyTorch 2.6+ 默认开启的 weights_only=True 限制"""
    try: return torch.load(path, map_location=device)
    except Exception: return torch.load(path, map_location=device, weights_only=False)

# ---------------------------------------------------------------------------
# Data Preparation & Regex Matching
# ---------------------------------------------------------------------------
def build_matchers(entries: Iterable[Tuple[str, str]], case_insensitive: bool = True, require_word_boundary: bool = True) -> List[Tuple[str, str, re.Pattern]]:
    """构建正则匹配器，处理边界符并按实体长度降序排列（优先匹配长实体）"""
    cleaned = [(str(lbl), str(txt).strip()) for lbl, txt in entries if str(txt).strip()]
    cleaned.sort(key=lambda x: (len(x[1]), x[1].lower()), reverse=True) 
    
    flags = re.IGNORECASE if case_insensitive else 0
    matchers = []
    for label, surface in cleaned:
        escaped = re.escape(surface)
        if require_word_boundary:
            # 动态决定是否添加单词边界零宽断言，防止标点符号导致匹配失败
            left = r"(?<![A-Za-z0-9])" if re.match(r"[A-Za-z0-9]", surface[0]) else ""
            right = r"(?![A-Za-z0-9])" if re.match(r"[A-Za-z0-9]", surface[-1]) else ""
            pattern = left + escaped + right
        else:
            pattern = escaped
        matchers.append((label, surface, re.compile(pattern, flags)))
    return matchers

def find_non_overlapping_spans(text: str, matchers: Sequence[Tuple[str, str, re.Pattern]]) -> List[Span]:
    """贪心匹配文本中的无重叠实体边界"""
    taken = [False] * len(text)
    spans = []
    for label, _, pat in matchers:
        for m in pat.finditer(text):
            start, end = m.span()
            if start < end and not any(taken[start:end]):
                taken[start:end] = [True] * (end - start)
                spans.append(Span(start, end, label))
    return sorted(spans, key=lambda s: s.start)

def spans_to_bio_chars(text: str, spans: Sequence[Span]) -> List[str]:
    """将实体的 span (启停位置) 转换为字符级的 BIO 标签序列"""
    tags = ["O"] * len(text)
    for sp in spans:
        if 0 <= sp.start < sp.end <= len(text):
            tags[sp.start] = f"B-{sp.label}"
            for i in range(sp.start + 1, sp.end): tags[i] = f"I-{sp.label}"
    return tags

def bio_to_spans(tags: Sequence[str]) -> List[Span]:
    """将 BIO 序列解析回 Span。包含容错逻辑：遇到非法的 I 标签会当作 B 标签处理"""
    spans = []
    start, label = None, None
    for i, t in enumerate(tags + ["O"]):
        if t.startswith("B-") or t == "O" or (start is not None and t != f"I-{label}"):
            if start is not None:
                spans.append(Span(start, i, label))
                start, label = None, None
        if t.startswith("B-"):
            start, label = i, t[2:]
    return spans

def smart_segment(text: str, tags: Sequence[str], max_len: int, min_len: int = 64) -> List[Tuple[str, List[str]]]:
    """智能长文本切分：尽量在空白字符处截断，避免将一个单词或实体劈成两半"""
    if len(text) <= max_len: return [(text, list(tags))]
    out = []
    start, n = 0, len(text)
    while start < n:
        if n - start <= max_len:
            out.append((text[start:], list(tags[start:])))
            break
        cut = start + max_len
        best = next((i for i in range(cut, max(start + min_len, start), -1) if text[i - 1].isspace()), cut)
        out.append((text[start:best], list(tags[start:best])))
        start = best
    return out

# ---------------------------------------------------------------------------
# Vocab & Dataset
# ---------------------------------------------------------------------------
class Vocab:
    """通用的词汇表类，兼容字符(Char)与标签(Tag)的编码解码"""
    def __init__(self, itos: List[str], unk_token: str = "<UNK>", pad_token: str = "<PAD>"):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.unk_id = self.stoi.get(unk_token, 0)
        self.pad_id = self.stoi.get(pad_token, 0)

    def encode(self, items: Sequence[str]) -> List[int]:
        return [self.stoi.get(x, self.unk_id) for x in items]

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.itos[i] for i in ids]

    def to_dict(self) -> dict: return {"itos": self.itos}
    @classmethod
    def from_dict(cls, d: dict) -> "Vocab": return cls(d["itos"])

class NerJsonlDataset(Dataset):
    def __init__(self, samples: List[dict], char_vocab: Vocab, tag_vocab: Vocab):
        self.samples = samples
        self.char_vocab = char_vocab
        self.tag_vocab = tag_vocab

    def __len__(self) -> int: return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "x": torch.tensor(self.char_vocab.encode(s["text"]), dtype=torch.long),
            "y": torch.tensor(self.tag_vocab.encode(s["tags"]), dtype=torch.long),
            **s
        }

    @staticmethod
    def collate_fn(batch: List[dict], pad_id: int) -> dict:
        """动态 Padding：按照当前 Batch 中最长的序列进行补齐，节省显存"""
        lengths = torch.tensor([len(b["x"]) for b in batch], dtype=torch.long)
        max_len = int(lengths.max().item())
        xs, ys = torch.full((len(batch), max_len), pad_id, dtype=torch.long), torch.zeros((len(batch), max_len), dtype=torch.long)
        mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        
        for i, b in enumerate(batch):
            L = len(b["x"])
            xs[i, :L], ys[i, :L], mask[i, :L] = b["x"], b["y"], True
        return {"x": xs, "y": ys, "mask": mask, "lengths": lengths, "texts": [b["text"] for b in batch], "gold_tags": [b["tags"] for b in batch]}

def create_dataloader(samples, c_vocab, t_vocab, batch_size, shuffle=False):
    ds = NerJsonlDataset(samples, c_vocab, t_vocab)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda b: ds.collate_fn(b, c_vocab.pad_id))

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def log_sum_exp(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """数值稳定的 log-sum-exp 实现，防止指数溢出"""
    max_score = tensor.max(dim, keepdim=True)[0]
    return max_score.squeeze(dim) + torch.log(torch.sum(torch.exp(tensor - max_score), dim))

class CRF(nn.Module):
    """纯手写的线性链条件随机场 (Linear-chain CRF)，支持 Batch 化计算"""
    def __init__(self, num_tags: int):
        super().__init__()
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """计算正确标签序列的对数似然 (Log-Likelihood)"""
        # 计算当前正确路径的得分 (Numerator)
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0, :].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for t in range(1, emissions.shape[1]):
            emit_t = emissions[:, t, :].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            trans_t = self.transitions[tags[:, t - 1], tags[:, t]]
            score += (emit_t + trans_t) * mask[:, t]
        score += self.end_transitions[tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)]
        
        # 使用前向算法(Forward Algorithm)计算所有可能路径的总得分，即配分函数 (Partition Function / Denominator)
        part_score = self.start_transitions + emissions[:, 0]
        for t in range(1, emissions.shape[1]):
            next_score = log_sum_exp(part_score.unsqueeze(1) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(2), dim=2)
            part_score = torch.where(mask[:, t].unsqueeze(1), next_score, part_score)
        log_den = log_sum_exp(part_score + self.end_transitions, dim=1)
        return score - log_den

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """使用 Viterbi 算法解码出得分最高的标签序列"""
        score = self.start_transitions + emissions[:, 0]
        history = []
        for t in range(1, emissions.shape[1]):
            next_score, indices = (score.unsqueeze(2) + self.transitions.unsqueeze(0)).max(1)
            score = torch.where(mask[:, t].unsqueeze(1), next_score + emissions[:, t], score)
            history.append(indices)

        score += self.end_transitions
        best_tags = score.max(1)[1]
        
        # 回溯寻找最优路径
        paths = []
        for i, end in enumerate(mask.sum(1).long() - 1):
            tag = best_tags[i].item()
            path = [tag]
            for t in range(end - 1, 0, -1):
                tag = history[t - 1][i][tag].item()
                path.append(tag)
            paths.append(path[::-1])
        return paths

class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size: int, num_tags: int, emb_dim: int=64, hid_dim: int=128, layers: int=1, drop: float=0.1, pad_id: int=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        # 使用 batch_first=True 简化维度操作
        self.lstm = nn.LSTM(emb_dim, hid_dim//2, layers, dropout=drop if layers>1 else 0.0, bidirectional=True, batch_first=True)
        self.dropout, self.fc, self.crf = nn.Dropout(drop), nn.Linear(hid_dim, num_tags), CRF(num_tags)

    def _emissions(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """通过 BiLSTM 提取特征并映射到标签空间"""
        # pack_padded_sequence 避免将 padding 符纳入 LSTM 的隐状态计算
        packed = nn.utils.rnn.pack_padded_sequence(self.embedding(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = nn.utils.rnn.pad_packed_sequence(self.lstm(packed)[0], batch_first=True)
        return self.fc(self.dropout(out))

    def loss(self, x, y, mask, lengths): return -self.crf(self._emissions(x, lengths), y, mask).mean()
    @torch.no_grad()
    def decode(self, x, mask, lengths): return self.crf.decode(self._emissions(x, lengths), mask)

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_prepare(args: argparse.Namespace) -> None:
    """弱监督数据准备：利用 Schema 词典对原始语料进行精确匹配标注"""
    in_p, out_p = resolve_path(args.input), resolve_path(args.output)
    base_matchers = build_matchers([(lbl, n) for lbl, names in SCHEMA_SEEDS.items() for n in names], not args.case_sensitive, not args.no_word_boundary)

    out_rows = []
    for row in iter_jsonl(in_p):
        text = str(row.get("text", ""))
        if not text: continue
        
        matchers = base_matchers
        # 若开启了动态种子，额外将每行语料的来源标题(title)作为高优匹配目标
        if not args.no_row_seeds and row.get("seed_label"):
            extra = [(row["seed_label"], x) for x in (row.get("seed_title"), row.get("title")) if x]
            if extra: matchers = build_matchers(extra, not args.case_sensitive, not args.no_word_boundary) + matchers

        tags = spans_to_bio_chars(text, find_non_overlapping_spans(text, matchers))
        for i, (seg_text, seg_tags) in enumerate(smart_segment(text, tags, args.max_len, args.min_len)):
            out_rows.append({"id": f"{row.get('doc_id', '')}__seg{i:03d}", "text": seg_text, "tags": seg_tags})

    write_jsonl(out_rows, out_p)
    print(f"[DONE] Prepared {len(out_rows)} labeled segments -> {out_p}")

def evaluate(model, loader, t_vocab, device) -> Tuple[float, float, float]:
    """计算基于 Span 的 Micro-P, R, F1 (采用了容错型解析)"""
    model.eval()
    tp, fp, fn = 0, 0, 0
    with torch.no_grad():
        for b in loader:
            preds = model.decode(b["x"].to(device), b["mask"].to(device), b["lengths"].to(device))
            for i, L in enumerate(b["lengths"].tolist()):
                g = {(s.start, s.end, s.label) for s in bio_to_spans(b["gold_tags"][i][:L])}
                p = {(s.start, s.end, s.label) for s in bio_to_spans(t_vocab.decode(preds[i][:L]))}
                tp += len(g & p); fp += len(p - g); fn += len(g - p)
    
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r)) if p + r else 0.0

def cmd_train(args: argparse.Namespace) -> None:
    """训练核心流程"""
    set_seed(args.seed)
    device, model_dir = get_device(), resolve_path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    samples = [{"id": r.get("id", ""), "text": r["text"], "tags": r["tags"]} for r in iter_jsonl(resolve_path(args.data))]
    random.shuffle(samples)
    n_tr, n_dv = int(len(samples) * args.train_ratio), int(len(samples) * args.dev_ratio)
    train_s, dev_s, test_s = samples[:n_tr], samples[n_tr:n_tr+n_dv], samples[n_tr+n_dv:]

    # 基于训练集构建字符词表
    c_freq = {}
    for s in train_s: 
        for c in s["text"]: c_freq[c] = c_freq.get(c, 0) + 1
    itos_c = ["<PAD>", "<UNK>"] + [c for c, _ in sorted(c_freq.items(), key=lambda x: -x[1]) if c not in ("<PAD>", "<UNK>")]
    
    c_vocab = Vocab(itos_c[:args.max_vocab_size])
    t_vocab = Vocab(["O"] + [f"{px}-{l}" for l in SCHEMA_SEEDS for px in "BI"])

    with (model_dir / "char_vocab.json").open("w") as f: json.dump(c_vocab.to_dict(), f)
    with (model_dir / "tag_vocab.json").open("w") as f: json.dump(t_vocab.to_dict(), f)

    train_dl = create_dataloader(train_s, c_vocab, t_vocab, args.batch_size, shuffle=True)
    dev_dl = create_dataloader(dev_s, c_vocab, t_vocab, args.batch_size)
    test_dl = create_dataloader(test_s, c_vocab, t_vocab, args.batch_size)

    model = BiLSTMCRF(len(c_vocab.itos), len(t_vocab.itos), args.embedding_dim, args.hidden_dim, args.num_layers, args.dropout, c_vocab.pad_id).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1, best_path = -1.0, model_dir / "model.pt"
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum, it = 0.0, tqdm(train_dl, desc=f"Ep {ep}/{args.epochs}") if tqdm else train_dl
        
        for b in it:
            loss = model.loss(b["x"].to(device), b["y"].to(device), b["mask"].to(device), b["lengths"].to(device))
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); optimizer.step()
            loss_sum += loss.item()
            if tqdm: it.set_postfix(loss=f"{loss.item():.4f}")

        dp, dr, df1 = evaluate(model, dev_dl, t_vocab, device)
        print(f"[Epoch {ep}] loss={loss_sum/len(train_dl):.4f} | Dev P: {dp:.3f} R: {dr:.3f} F1: {df1:.3f}")
        if df1 > best_f1:
            best_f1 = df1
            torch.save({"state_dict": model.state_dict(), "config": _safe_ckpt_config(args)}, best_path)

    # 训练结束后加载 Best Model 并测试
    model.load_state_dict(_torch_load_compat(best_path, device)["state_dict"])
    tp, tr, tf1 = evaluate(model, test_dl, t_vocab, device)
    print(f"[Final Test] P: {tp:.4f} R: {tr:.4f} F1: {tf1:.4f}")

def cmd_predict(args: argparse.Namespace) -> None:
    """预测入口：支持直接传入字符串，或读取 JSONL 文件"""
    device, model_dir = get_device(), resolve_path(args.model_dir)
    with (model_dir/"char_vocab.json").open() as f1, (model_dir/"tag_vocab.json").open() as f2:
        c_vocab, t_vocab = Vocab.from_dict(json.load(f1)), Vocab.from_dict(json.load(f2))

    ckpt = _torch_load_compat(model_dir / "model.pt", device)
    cfg = ckpt.get("config", {})
    model = BiLSTMCRF(len(c_vocab.itos), len(t_vocab.itos), cfg.get("embedding_dim", 64), cfg.get("hidden_dim", 128), cfg.get("num_layers", 1), cfg.get("dropout", 0.1), c_vocab.pad_id).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    texts = [args.text] if args.text else [r["text"] for r in iter_jsonl(resolve_path(args.input)) if r.get("text")]
    for t in texts:
        x, m, l = torch.tensor([c_vocab.encode(t)], device=device), torch.ones((1, len(t)), dtype=torch.bool, device=device), torch.tensor([len(t)], device=device)
        tags = t_vocab.decode(model.decode(x, m, l)[0])
        print(json.dumps({"text": t, "spans": [{"start": s.start, "end": s.end, "label": s.label, "text": t[s.start:s.end]} for s in bio_to_spans(tags)]}, ensure_ascii=False))

# ---------------------------------------------------------------------------
# CLI Entry
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BiLSTM-CRF NER (char-level) with weak supervision")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Prepare command
    p_p = sub.add_parser("prepare")
    for arg, val in [("--input", "data/raw/turing_schema_corpus.jsonl"), ("--output", "data/intermediate/ner_char_bio.jsonl")]: p_p.add_argument(arg, default=val)
    p_p.add_argument("--max-len", type=int, default=1024)
    p_p.add_argument("--min-len", type=int, default=64)
    p_p.add_argument("--case-sensitive", action="store_true")
    p_p.add_argument("--no-word-boundary", action="store_true")
    p_p.add_argument("--no-row-seeds", action="store_true")
    p_p.set_defaults(func=cmd_prepare)

    # Train command
    p_t = sub.add_parser("train")
    for arg, val in [("--data", "data/intermediate/ner_char_bio.jsonl"), ("--model-dir", "data/output/ner_bilstm_crf")]: p_t.add_argument(arg, default=val)
    for arg, type_, val in [("--seed", int, 42), ("--train-ratio", float, 0.8), ("--dev-ratio", float, 0.1), ("--embedding-dim", int, 64), ("--hidden-dim", int, 128), ("--num-layers", int, 1), ("--dropout", float, 0.1), ("--max-vocab-size", int, 5000), ("--batch-size", int, 32), ("--epochs", int, 8), ("--lr", float, 1e-3), ("--grad-clip", float, 5.0)]: p_t.add_argument(arg, type=type_, default=val)
    p_t.set_defaults(func=cmd_train)

    # Predict command
    p_pr = sub.add_parser("predict")
    for arg, val in [("--model-dir", "data/output/ner_bilstm_crf"), ("--text", ""), ("--input", "")]: p_pr.add_argument(arg, default=val)
    p_pr.set_defaults(func=cmd_predict)
    
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)