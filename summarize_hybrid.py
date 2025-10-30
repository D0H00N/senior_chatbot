import os, re, json, argparse
import numpy as np, pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import ast

# -------------------------
# 프롬프트 생성 (chat 템플릿 있으면 role 기반, 없으면 문자열 프롬프트)
# -------------------------
def render_prompt(tokenizer, text: str) -> str:
    template_user = (
        "- 아래 글을 5~7문장으로 요약하세요.\n"
        "- 처음 1~2문장은 문제/배경, 이후는 해결 방법과 핵심 절차를 기사 문체로 서술\n"
        "- 불필요한 잡설/반복 금지\n"
        "- 입력에 없는 사실/주제를 새로 추가하지 마세요(환상 금지)\n"
        "- 입력에 등장한 용어/개체명만 사용하세요\n"
        "- 입력 본문과 직접 관련된 문장만 사용하세요\n"
        "- 100프로 한국어만 사용하고 띄어쓰기와 문법에 따라 출력하세요\n\n"
        f"{text}"
    )

    messages = [
        {"role": "system", "content": "너는 한국어 문법 교정을 마스터한 전문 뉴스 요약 기자다."},
        {"role": "user", "content": template_user}
    ]

    chat_tmpl = getattr(tokenizer, "chat_template", None)
    if chat_tmpl and str(chat_tmpl).strip():
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # 폴백: 일반 문자열 프롬프트
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

# -------------------------
# 프롬프트 에코 제거
# -------------------------
SPECIAL_BLOCK_RE = re.compile(
    r"(###\s*(답변|요약|질문|명령어|입력|맥락|응답)\s*:?.*?$)|"  # ### 요약/질문/명령어/입력/맥락/응답
    r"(<\|im_start\|>.*?<\|im_end\|>)|"                   # <|im_start|> ~ <|im_end|>
    r"^\s*(system|user|assistant)\b.*$",                  # role 라인
    flags=re.IGNORECASE | re.DOTALL | re.MULTILINE
)

def strip_prompt_echo(text: str, min_sents=5, max_sents=7) -> str:
    # 1) "### 요약:" 이후만 남기기
    cut = text.find("### 요약:")
    if cut != -1:
        text = text[cut + len("### 요약:"):]

    # 2) 특수 블록 제거
    text = SPECIAL_BLOCK_RE.sub("", text)

    # 3) 문장 단위로 분리
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s.strip() for s in sents if s.strip()]

    # 4) 문장 개수 제약 (짧으면 그대로, 길면 자르기)
    if len(sents) > max_sents:
        sents = sents[:max_sents]

    # 5) 다시 합치기
    return " ".join(sents).strip()
# -------------------------
# 모델 로딩 (SDPA)
# -------------------------
def make_pipeline(model_id: str, device: int = 0):
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        attn_implementation="sdpa",
    )
    gen = pipeline("text-generation", model=model, tokenizer=tok, device=device)
    return gen

# -------------------------
# 요약
# -------------------------
def llm_summarize(gen, text: str, max_new_tokens=220) -> str:
    if not str(text).strip():
        return "(내용 없음)"
    tok = gen.tokenizer
    prompt = render_prompt(tok, text[:5000])
    out = gen(
        prompt, max_new_tokens=max_new_tokens, do_sample=False, 
        return_full_text=False,  no_repeat_ngram_size=3,    
        repetition_penalty=1.12 )[0]
    raw = out.get("generated_text", "") if isinstance(out, dict) else str(out)
    return strip_prompt_echo(raw)

# -------------------------
# 임베딩 파싱 / Retrieval
# -------------------------
def parse_embedding(x):
    if isinstance(x, str):
        try:
            return np.array(ast.literal_eval(x), dtype=np.float32)
        except Exception:
            return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.array(x, dtype=np.float32)
    return None

def retrieve_topk(texts, embs, top_k=12):
    valid = [(t, e) for t, e in zip(texts, embs) if e is not None]
    if not valid:
        return texts[:top_k]
    tex, X = zip(*valid)
    X = np.vstack(X).astype(np.float32)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-6, None)
    q = X.mean(axis=0)
    sims = X @ q
    idx = np.argsort(-sims)[:top_k]
    # 원문 흐름 보존하려면 sorted(idx) 대신 idx 그대로도 가능 (여긴 정렬 유지)
    return [tex[i] for i in sorted(idx)]

# -------------------------
# I/O
# -------------------------
def load_any(path: str) -> pd.DataFrame:
    # 경로 정리 + 폴더(Parquet dataset) 우선
    path = os.path.normpath(str(path).strip().strip('"').strip("'"))
    if os.path.isdir(path):
        return pd.read_parquet(path, engine="pyarrow")
    lo = path.lower()
    if lo.endswith(".parquet"):
        return pd.read_parquet(path, engine="pyarrow")
    if lo.endswith(".json"):
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            with open(path, "r", encoding="utf-8") as f:
                return pd.DataFrame(json.load(f))
    if lo.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"지원하지 않는 형식: {path}")

def save_video_parquet(folder, vid, summary):
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, f"{vid}.parquet")
    # 오타 방지: 중괄호 제거
    out = os.path.join(folder, f"{vid}.parquet")
    url = f"https://youtu.be/{vid}"
    pd.DataFrame(
        {"video_id": [vid], "summary_text": [summary], "video_url": [url]}
    ).to_parquet(out, index=False)

# -------------------------
# 주 루프
# -------------------------
def save_any(df, output, model_id, device, max_new_tokens, top_k):
    gen = make_pipeline(model_id, device)
    groups = df.sort_values(["video_id", "chunk_id"]).groupby("video_id", dropna=False)

    # embedding 컬럼 있으면 사용
    embed_col = next((c for c in ["embedding", "emb", "vector", "embeddings"] if c in df.columns), None)

    for vid, g in tqdm(groups, desc="Videos", unit="video"):
        texts = [str(x or "") for x in g["text"].tolist()]
        embs = [parse_embedding(x) for x in g[embed_col].tolist()] if embed_col else [None] * len(texts)
        picked = retrieve_topk(texts, embs, top_k=top_k)
        context = "\n".join(picked)
        summary = llm_summarize(gen, context, max_new_tokens)
        summary = strip_prompt_echo(summary)
        save_video_parquet(output, str(vid), summary)

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Minimal RAG + LLM Summarizer (render_prompt 통합)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    # 한국어 전용 공개 모델 권장: beomi/KoAlpaca-Polyglot-12.8B)
    ap.add_argument("--model", default="MLP-KTLim/llama-3-Korean-Bllossom-8B")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=220)
    ap.add_argument("--top_k", type=int, default=12)
    args = ap.parse_args()

    df = load_any(args.input)
    # 필수 컬럼 확인
    if not {"video_id", "chunk_id", "text"}.issubset(df.columns):
        raise ValueError("반드시 video_id, chunk_id, text 컬럼 필요")

    save_any(df, args.output, args.model, args.device, args.max_new_tokens, args.top_k)

if __name__ == "__main__":
    main()
