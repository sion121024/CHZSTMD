"""from-scratch 한국어 대화 Transformer — Kaggle GPU 학습 스크립트(payload).

이 파일은 Kaggle 커널로 푸시되어 GPU에서 실행된다(MCP의 push_kernel).
사전학습 모델/체크포인트를 *적재하지 않는다*. GPT형 디코더 Transformer 구조를
여기서 직접 정의하고, 대화 말뭉치(뉴스·댓글 아님)로 처음부터 학습한다.

데이터: /kaggle/input 아래 마운트된 대화 데이터셋에서 (사용자 발화, 응답) 턴을
추출해 멀티턴 시퀀스로 만든다. jsonl/json/csv/txt의 흔한 대화 포맷을 처리한다.

출력: /kaggle/working 에 ckpt.pt(가중치+설정) 와 spm.model(토크나이저) 저장.
       MCP의 kaggle_kernel_output 으로 회수해 아바타에 연결한다.

로컬 검증: `python train_dialogue.py --smoke` (작은 모델 + 합성데이터, GPU 불필요).
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== 모델 (처음부터 정의) ============================== #
class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_head: int, dropout: float):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.dim = dim

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        h = self.n_head
        q = q.view(B, T, h, C // h).transpose(1, 2)
        k = k.view(B, T, h, C // h).transpose(1, 2)
        v = v.view(B, T, h, C // h).transpose(1, 2)
        # PyTorch 내장 causal attention (효율적). 없으면 수동 폴백.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.drop.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, dim, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, ctx, dim=384, n_layer=6, n_head=6, dropout=0.1):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(ctx, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dim, n_head, dropout) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.tok.weight = self.head.weight        # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos)[None, :, :])
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=80, temperature=0.9, top_k=40, eos=None):
        for _ in range(max_new):
            idx_cond = idx[:, -self.ctx:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-5, temperature)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos is not None and int(nxt) == eos:
                break
        return idx


# ===================== 대화 데이터 로딩(뉴스/댓글 아님) ==================== #
USER_TAG, BOT_TAG = "<사용자>", "<하루>"

# HF/파일 데이터에 접근 못 할 때 쓰는 최소 내장 시드(제대로 된 대화체, 뉴스/댓글 아님).
# 실제 유창함은 HF 대화셋으로 학습할 때 나온다 — 이건 파이프라인이 절대 깨지지 않게
# 하는 안전장치다.
_SEED = [
    [("u", "하루야 안녕"), ("b", "안녕! 오늘도 와줘서 고마워")],
    [("u", "오늘 뭐 할 거야?"), ("b", "신작 게임 한 번 깨보려고! 같이 보자")],
    [("u", "이 게임 어려워?"), ("b", "처음엔 좀 헷갈리는데 금방 적응돼. 해볼 만해")],
    [("u", "방금 진짜 잘했다"), ("b", "헤헤 봤어? 나 좀 늘었지?")],
    [("u", "배고프지 않아?"), ("b", "조금! 이판만 깨고 같이 뭐 먹을까 고민 중이야")],
    [("u", "노래도 할 줄 알아?"), ("b", "응 한 소절 정도는! 다음에 들려줄게")],
    [("u", "왜 이렇게 텐션이 높아 ㅋㅋ"), ("b", "방송이 제일 재밌으니까! 너희 덕분이지")],
    [("u", "조심해 적이 온다"), ("b", "오 고마워! 바로 막을게, 집중집중")],
    [("u", "오늘 몇 시까지 해?"), ("b", "두 시간 정도 더 할 생각이야. 천천히 놀다 가")],
    [("u", "수고했어 오늘"), ("b", "너도 고생했어! 내일 또 보자, 잘 자")],
    [("u", "하루 기분 어때?"), ("b", "완전 좋아! 오늘따라 손이 잘 풀려")],
    [("u", "그 보스 어떻게 잡아?"), ("b", "패턴 외우는 게 핵심이야. 한 번 보여줄게")],
    [("u", "방금 거 깜짝 놀랐잖아"), ("b", "헉 나도! 갑자기 튀어나오니까 심장 떨어질 뻔")],
    [("u", "물 좀 마시고 해"), ("b", "맞다, 챙겨줘서 고마워. 한 모금 하고 갈게")],
    [("u", "구독했어!"), ("b", "우와 진짜? 완전 고마워, 덕분에 힘난다!")],
    [("u", "오늘 옷 예쁘다"), ("b", "헤헤 알아봐 줘서 고마워! 오늘 좀 신경 썼지")],
    [("u", "이거 어떻게 생각해?"), ("b", "음, 내 생각엔 충분히 해볼 만한 것 같아")],
    [("u", "졌네 ㅠㅠ"), ("b", "괜찮아! 다음 판 만회하면 되지, 가보자")],
    [("u", "내일도 방송해?"), ("b", "응 내일도 같은 시간에! 꼭 와줘")],
    [("u", "하루 최고야"), ("b", "에헤헤 그렇게 말해주니 너무 기분 좋다")],
    [("u", "여기서 어디로 가?"), ("b", "오른쪽 길로 가보자. 뭔가 있을 것 같아")],
    [("u", "템 좋은 거 나왔어?"), ("b", "오 이거 봐봐! 완전 대박템 떴어")],
    [("u", "좀 쉬었다 하자"), ("b", "그래 잠깐 쉴까? 너희도 스트레칭 한 번 하고")],
    [("u", "그 캐릭터 귀엽다"), ("b", "그치그치! 나도 얘 디자인 완전 좋아해")],
    [("u", "집중 잘 안 돼?"), ("b", "조금? 그래도 너희랑 떠들면서 하니까 괜찮아")],
    [("u", "방금 그 점프 미쳤다"), ("b", "내가 봐도 좀 멋졌어 ㅋㅋ 또 해볼까?")],
    [("u", "오늘 컨디션 좋네"), ("b", "맞아! 왠지 다 잘 풀리는 날이야")],
    [("u", "다음 게임 뭐 할까?"), ("b", "음 너희가 골라줘! 채팅에 추천 적어줘")],
    [("u", "고생 많았어"), ("b", "고마워, 너희가 있어서 끝까지 할 수 있었어")],
    [("u", "안녕 처음 왔어"), ("b", "오 어서 와! 편하게 놀다 가, 반가워")],
]


def _seed_corpus(reps: int = 60) -> str:
    out = []
    for _ in range(reps):
        for turns in _SEED:
            chunk = [f"{(USER_TAG if i % 2 == 0 else BOT_TAG)} {t[1]}"
                     for i, t in enumerate(turns)]
            out.append("\n".join(chunk) + "\n<eos>\n")
    return "\n".join(out)


def _iter_dialogues(paths):
    """여러 포맷에서 (turns) 리스트를 뽑는다. turns=[(speaker, text), ...]."""
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".jsonl", ".ndjson"):
                for line in open(path, encoding="utf-8"):
                    line = line.strip()
                    if line:
                        yield from _extract(json.loads(line))
            elif ext == ".json":
                data = json.load(open(path, encoding="utf-8"))
                items = data if isinstance(data, list) else [data]
                for it in items:
                    yield from _extract(it)
            elif ext == ".csv":
                import csv
                r = csv.DictReader(open(path, encoding="utf-8"))
                for row in r:
                    turns = []
                    for col in ("Q", "A", "question", "answer", "utterance", "response",
                                "사용자", "응답", "user", "bot"):
                        if col in row and row[col]:
                            turns.append(("?", row[col]))
                    if len(turns) >= 2:
                        yield turns
            elif ext == ".txt":
                buf = []
                for line in open(path, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        if len(buf) >= 2:
                            yield [("?", t) for t in buf]
                        buf = []
                    else:
                        buf.append(line)
                if len(buf) >= 2:
                    yield [("?", t) for t in buf]
        except Exception as e:
            print(f"[warn] {path} 파싱 실패: {e}", file=sys.stderr)


def _extract(obj):
    """대화 객체 하나에서 turns를 뽑는다(흔한 키들)."""
    for key in ("conversations", "dialogue", "utterances", "turns", "messages"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], list):
            turns = []
            for t in obj[key]:
                if isinstance(t, dict):
                    txt = t.get("value") or t.get("text") or t.get("utterance") or t.get("content")
                    spk = t.get("from") or t.get("speaker") or t.get("role") or "?"
                    if txt:
                        turns.append((str(spk), str(txt)))
                elif isinstance(t, str):
                    turns.append(("?", t))
            if len(turns) >= 2:
                yield turns
            return
    if isinstance(obj, dict):
        q = obj.get("Q") or obj.get("question") or obj.get("input")
        a = obj.get("A") or obj.get("answer") or obj.get("output") or obj.get("response")
        if q and a:
            yield [("user", str(q)), ("bot", str(a))]


def build_corpus_hf(dataset: str, split: str = "train", max_dialogues: int = 0):
    """HuggingFace 대화 데이터셋에서 코퍼스를 만든다(뉴스/댓글 아님, 멀티턴).

    Kaggle 커널엔 `datasets`가 보통 설치돼 있다(없으면 pip install datasets).
    ShareGPT류(example['conversations']=[{from,value},...]) 등 흔한 대화 스키마를
    _extract 로 처리한다.
    """
    try:
        from datasets import load_dataset  # Kaggle 런타임에 존재
        ds = load_dataset(dataset, split=split)
    except Exception as e:
        print(f"[warn] HF '{dataset}' 로드 실패({e}) → 다른 소스로 폴백", file=sys.stderr)
        return ""
    lines, n = [], 0
    for ex in ds:
        for turns in _extract(ex):
            chunk = [f"{(USER_TAG if i % 2 == 0 else BOT_TAG)} {t[1].strip()}"
                     for i, t in enumerate(turns)]
            lines.append("\n".join(chunk) + "\n<eos>\n")
            n += 1
        if max_dialogues and n >= max_dialogues:
            break
    print(f"[data] HF '{dataset}' 대화 {n}건 로드")
    return "\n".join(lines)


def build_corpus(paths, max_dialogues=0):
    """대화들을 역할 태그가 붙은 텍스트 한 덩어리로 만든다."""
    lines = []
    n = 0
    for turns in _iter_dialogues(paths):
        chunk = []
        for i, (_spk, text) in enumerate(turns):
            tag = USER_TAG if i % 2 == 0 else BOT_TAG
            chunk.append(f"{tag} {text.strip()}")
        lines.append("\n".join(chunk) + "\n<eos>\n")
        n += 1
        if max_dialogues and n >= max_dialogues:
            break
    print(f"[data] 대화 {n}건 로드")
    return "\n".join(lines)


# ===================== 토크나이저 ======================================== #
class ByteTok:
    """의존성 0 폴백 토크나이저(스모크용). 바이트 단위."""
    vocab_size = 259
    eos = 256
    def encode(self, s): return list(s.encode("utf-8"))
    def decode(self, ids): return bytes(b for b in ids if b < 256).decode("utf-8", "ignore")


def train_spm(corpus_text, model_prefix, vocab_size):
    """SentencePiece BPE 학습. 코퍼스가 작아 vocab이 너무 크면 자동으로 줄여 재시도."""
    import sentencepiece as spm
    with open("_corpus.txt", "w", encoding="utf-8") as f:
        f.write(corpus_text)
    v = vocab_size
    while True:
        try:
            spm.SentencePieceTrainer.train(
                input="_corpus.txt", model_prefix=model_prefix, vocab_size=v,
                model_type="bpe", character_coverage=0.9995,
                user_defined_symbols=[USER_TAG, BOT_TAG, "<eos>"],
                pad_id=0, unk_id=1, bos_id=2, eos_id=3)
            break
        except Exception as e:
            import re
            m = re.search(r"<=\s*(\d+)", str(e))   # spm이 알려주는 최대 vocab
            v = int(m.group(1)) if m else v // 2
            if v < 64:
                raise
            print(f"[warn] vocab 축소 재시도 → {v}", file=sys.stderr)
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    return sp, sp.get_piece_size()


# ===================== 학습 루프 ========================================= #
def get_batch(data, ctx, bs, device):
    ix = torch.randint(len(data) - ctx - 1, (bs,))
    x = torch.stack([data[i:i + ctx] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + ctx] for i in ix])
    return x.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input", help="대화 데이터 루트(Kaggle 마운트)")
    ap.add_argument("--glob", default="**/*.jsonl,**/*.json,**/*.csv,**/*.txt")
    # 권장: 제대로 된 멀티턴 대화 데이터를 HF에서 받는다(뉴스/댓글 아님).
    ap.add_argument("--hf_dataset", default="junelee/sharegpt_deepl_ko",
                    help="HuggingFace 대화 데이터셋. ''로 비우면 --data의 Kaggle 파일 사용.")
    ap.add_argument("--hf_split", default="train")
    ap.add_argument("--max_dialogues", type=int, default=0, help="0=전체")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--smoke", action="store_true", help="작은 모델+합성데이터 검증")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # GPU가 잡혀도 torch 빌드와 안 맞으면(no kernel image) 실제 연산이 죽는다.
    # 작은 연산으로 점검하고, 실패하면 CPU로 폴백한다.
    if device == "cuda":
        try:
            _ = (torch.zeros(8, device="cuda") + 1.0).sum().item()
        except Exception as e:
            print(f"[warn] CUDA 사용 불가({e}) → CPU로 폴백", file=sys.stderr)
            device = "cpu"
    if device == "cpu" and not args.smoke:
        # CPU 폴백: 제한 시간 안에 끝나도록 모델/스텝 축소
        args.dim = min(args.dim, 192)
        args.layers = min(args.layers, 4)
        args.heads = min(args.heads, 4)
        args.ctx = min(args.ctx, 192)
        args.steps = min(args.steps, 1500)
        args.batch = min(args.batch, 16)
        print(f"[info] CPU 모드 — dim={args.dim} layers={args.layers} steps={args.steps}",
              file=sys.stderr)
    os.makedirs(args.out, exist_ok=True)

    # ---- 데이터 + 토크나이저 ----
    if args.smoke:
        corpus = (f"{USER_TAG} 안녕\n{BOT_TAG} 안녕! 반가워\n<eos>\n"
                  f"{USER_TAG} 뭐해\n{BOT_TAG} 게임하지 ㅋㅋ\n<eos>\n") * 200
        tok = ByteTok()
        ids = torch.tensor(tok.encode(corpus), dtype=torch.long)
        args.vocab, args.ctx, args.dim, args.layers, args.heads = 259, 64, 96, 2, 3
        args.steps, args.batch = 30, 8
        eos = tok.eos
        sp = None
    else:
        # 데이터 우선순위: HF 대화셋 → Kaggle 마운트 파일 → 내장 시드(절대 안 깨짐)
        corpus = ""
        if args.hf_dataset:
            corpus = build_corpus_hf(args.hf_dataset, args.hf_split, args.max_dialogues)
        if not corpus.strip():
            globs = []
            for g in args.glob.split(","):
                globs += glob.glob(os.path.join(args.data, g.strip()), recursive=True)
            if globs:
                corpus = build_corpus(globs)
        if not corpus.strip():
            print("[data] HF/파일 없음 → 내장 시드 대화로 학습(파이프라인 검증용).",
                  file=sys.stderr)
            corpus = _seed_corpus()
        sp, args.vocab = train_spm(corpus, os.path.join(args.out, "spm"), args.vocab)
        print(f"[tok] vocab={args.vocab}")
        eos = sp.piece_to_id("<eos>")
        ids = torch.tensor(sp.encode(corpus), dtype=torch.long)

    n = int(len(ids) * 0.98)
    train_d, val_d = ids[:n], ids[n:]
    print(f"[data] 토큰 {len(ids):,} (train {len(train_d):,}/val {len(val_d):,}), device={device}")

    model = GPT(args.vocab, args.ctx, args.dim, args.layers, args.heads).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[model] from-scratch GPT 파라미터 {nparam:,} (사전학습 없음)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * (0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        x, y = get_batch(train_d, args.ctx, args.batch, device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % max(1, args.steps // 20) == 0 or step == 1:
            print(f"  step {step}/{args.steps} loss {loss.item():.3f} "
                  f"lr {lr:.1e} {(time.time()-t0):.0f}s")

    # ---- 저장 ----
    ckpt = os.path.join(args.out, "ckpt.pt")
    torch.save({"model": model.state_dict(),
                "config": {"vocab": args.vocab, "ctx": args.ctx, "dim": args.dim,
                           "layers": args.layers, "heads": args.heads},
                "eos": eos,
                "tokenizer": "spm" if sp else "byte"}, ckpt)
    print(f"[save] {ckpt}")

    # ---- 생성 샘플 ----
    model.eval()
    prompt = f"{USER_TAG} 안녕\n{BOT_TAG}"
    enc = sp.encode(prompt) if sp else tok.encode(prompt)
    out = model.generate(torch.tensor([enc], device=device), max_new=40, eos=eos)[0].tolist()
    text = sp.decode(out) if sp else tok.decode(out)
    print("[sample]", text.replace("\n", " / "))


if __name__ == "__main__":
    main()
