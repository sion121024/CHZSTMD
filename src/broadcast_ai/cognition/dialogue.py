"""학습된 대화 모델 추론 어댑터.

Kaggle에서 from-scratch로 학습한 체크포인트(ckpt.pt + spm.model)를 불러와 한국어
응답을 생성한다. torch나 체크포인트가 없으면 import/생성 시 조용히 비활성화되어
Brain이 기존 Verbalizer(템플릿)로 폴백한다 — 코어는 의존성 0을 유지한다.

학습 측 구조(tools/mcp/kaggle_mcp/payload/train_dialogue.py)와 *동일한* GPT를 추론용
으로 가볍게 재정의한다(사전학습 모델 아님, 우리가 학습한 가중치만 적재).
"""

from __future__ import annotations

import os
from typing import Iterator

USER_TAG, BOT_TAG = "<사용자>", "<하루>"


def _torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


class _ByteTok:
    eos = 256
    def encode(self, s): return list(s.encode("utf-8"))
    def decode(self, ids): return bytes(b for b in ids if b < 256).decode("utf-8", "ignore")


class DialogueModel:
    """체크포인트 기반 응답 생성기. 없으면 available()=False."""

    def __init__(self, ckpt_dir: str) -> None:
        self.ok = False
        self.torch = _torch()
        if self.torch is None:
            return
        ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")
        if not os.path.exists(ckpt_path):
            return
        try:
            self._load(ckpt_dir, ckpt_path)
            self.ok = True
        except Exception as e:  # pragma: no cover
            print(f"[dialogue] 체크포인트 적재 실패 → Verbalizer 폴백: {e}")

    def available(self) -> bool:
        return self.ok

    def _load(self, ckpt_dir: str, ckpt_path: str) -> None:
        torch = self.torch
        ck = torch.load(ckpt_path, map_location="cpu")
        cfg = ck["config"]
        self.eos = ck.get("eos", 256)
        if ck.get("tokenizer") == "spm" and os.path.exists(os.path.join(ckpt_dir, "spm.model")):
            import sentencepiece as spm  # type: ignore
            self.sp = spm.SentencePieceProcessor(model_file=os.path.join(ckpt_dir, "spm.model"))
            self._enc = self.sp.encode
            self._dec = self.sp.decode
        else:
            t = _ByteTok()
            self._enc, self._dec, self.eos = t.encode, t.decode, t.eos
        self.model = _build_gpt(torch, cfg)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.ctx = cfg["ctx"]

    def reply(self, history: list[tuple[str, str]], user_text: str,
              max_new: int = 80, temperature: float = 0.9) -> str:
        """대화 이력 + 사용자 발화로 한 턴 응답을 생성한다."""
        torch = self.torch
        parts = []
        for i, (_role, text) in enumerate(history[-6:]):
            tag = USER_TAG if i % 2 == 0 else BOT_TAG
            parts.append(f"{tag} {text}")
        parts.append(f"{USER_TAG} {user_text}")
        parts.append(f"{BOT_TAG} ")
        prompt = "\n".join(parts)
        ids = self._enc(prompt)[-self.ctx:]
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = self.model.generate(x, max_new=max_new, temperature=temperature,
                                      eos=self.eos)[0].tolist()
        gen = self._dec(out[len(ids):])
        # 다음 화자 태그/eos 전까지만 응답으로 사용
        for stop in (USER_TAG, BOT_TAG, "<eos>"):
            if stop in gen:
                gen = gen.split(stop)[0]
        return gen.strip() or "음~ 잠깐만!"


# --------------------------------------------------------------------------- #
def _build_gpt(torch, cfg):
    import torch.nn as nn
    import torch.nn.functional as F

    class CSA(nn.Module):
        def __init__(self, dim, h):
            super().__init__()
            self.h = h
            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)

        def forward(self, x):
            B, T, C = x.shape
            q, k, v = self.qkv(x).split(C, dim=2)
            q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
            k = k.view(B, T, self.h, C // self.h).transpose(1, 2)
            v = v.view(B, T, self.h, C // self.h).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

    class Block(nn.Module):
        def __init__(self, dim, h):
            super().__init__()
            self.ln1 = nn.LayerNorm(dim); self.attn = CSA(dim, h)
            self.ln2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

        def forward(self, x):
            x = x + self.attn(self.ln1(x)); return x + self.mlp(self.ln2(x))

    class GPT(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.ctx = c["ctx"]
            self.tok = nn.Embedding(c["vocab"], c["dim"])
            self.pos = nn.Embedding(c["ctx"], c["dim"])
            self.blocks = nn.ModuleList([Block(c["dim"], c["heads"]) for _ in range(c["layers"])])
            self.lnf = nn.LayerNorm(c["dim"])
            self.head = nn.Linear(c["dim"], c["vocab"], bias=False)
            self.tok.weight = self.head.weight

        def forward(self, idx):
            T = idx.size(1)
            pos = torch.arange(T, device=idx.device)
            x = self.tok(idx) + self.pos(pos)[None]
            for b in self.blocks:
                x = b(x)
            return self.head(self.lnf(x))

        @torch.no_grad()
        def generate(self, idx, max_new=80, temperature=0.9, top_k=40, eos=None):
            for _ in range(max_new):
                logits = self(idx[:, -self.ctx:])[:, -1, :] / max(1e-5, temperature)
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
                idx = torch.cat([idx, nxt], dim=1)
                if eos is not None and int(nxt) == eos:
                    break
            return idx

    return GPT(cfg)
