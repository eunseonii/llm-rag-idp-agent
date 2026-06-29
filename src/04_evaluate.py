"""
04_evaluate.py

하이브리드 검색 파이프라인의 검색 품질을 Recall@k로 평가한다.

평가 흐름:
  1. JSONL 청크에서 랜덤 샘플링
  2. qwen3:8b로 질문 자동 생성 → eval_set.jsonl 저장
  3. hybrid_search()로 각 질문 검색
  4. Recall@1, @2, @4 계산 및 출력

파이프라인 위치: 03_hybrid_search.py → [현재] → 05_reranker_bench.py
"""

import json
import random
import time
import requests
from pathlib import Path
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import SparseVector


# ── 설정 ────────────────────────────────────────────────────
CHUNKS_DIR    = Path(__file__).parent.parent / "data" / "chunks"
EVAL_SET_PATH = Path(__file__).parent.parent / "data" / "eval_set.jsonl"
OLLAMA_URL      = "http://localhost:11434"
GRANITE_MODEL   = "granite-embedding:278m"
QWEN_MODEL      = "qwen3:8b"
BM25_MODEL_NAME = "Qdrant/bm25"
QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "kr_rules_study"

SAMPLE_SIZE     = 50   # 정답셋 질문 개수
CANDIDATE_K     = 20   # 검색 후보 개수
EVAL_K_LIST     = [1, 2, 4]  # Recall@k 측정 기준
RANDOM_SEED     = 42
RRF_K           = 60
TOP_K           = 4 

def embed_dense(query: str) -> list[float]:
    """Ollama Granite-embedding으로 쿼리 Dense 벡터 생성."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": GRANITE_MODEL, "prompt": query},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def embed_sparse(query: str, model: SparseTextEmbedding) -> SparseVector:
    """fastembed BM25로 쿼리 Sparse 벡터 생성."""
    result = list(model.embed([query]))[0]  # generator → 첫 번째 결과
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


# ────────────────────────────────────────────────────────────
# 2. 검색 함수
# ────────────────────────────────────────────────────────────

def search_dense(client: QdrantClient, vector: list[float], top_k: int) -> list[dict]:
    """Dense 벡터로 Qdrant 검색."""
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        using="dense",
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "id":       p.id,
            "score":    p.score,
            "text":     p.payload.get("text", ""),
            "source":   p.payload.get("source", ""),
            "page":     p.payload.get("page", -1),
            "chunk_id": p.payload.get("chunk_id", ""),
        }
        for p in response.points
    ]


def search_sparse(client: QdrantClient, vector: SparseVector, top_k: int) -> list[dict]:
    """Sparse(BM25) 벡터로 Qdrant 검색."""
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.SparseVector(
            indices=vector.indices,
            values=vector.values,
        ),
        using="sparse",
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "id":       p.id,
            "score":    p.score,
            "text":     p.payload.get("text", ""),
            "source":   p.payload.get("source", ""),
            "page":     p.payload.get("page", -1),
            "chunk_id": p.payload.get("chunk_id", ""),
        }
        for p in response.points
    ]


# ────────────────────────────────────────────────────────────
# 3. RRF 구현
# ────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = RRF_K,
    top_n: int = TOP_K,
    min_score: float = 0.01,
) -> list[dict]:
    """
    RRF(Reciprocal Rank Fusion) 직접 구현.

    각 검색기에서 받은 순위(rank)를 이용해 점수를 계산한다:
        rrf_score += 1 / (k + rank)   (rank는 1-based)

    두 검색기의 점수를 같은 문서끼리 합산한 뒤 내림차순 정렬.
    """
    scores: dict[str, float] = {}   # chunk_id → rrf 점수
    meta:   dict[str, dict]  = {}   # chunk_id → payload

    for rank, doc in enumerate(dense_results, start=1):
        cid = doc["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        meta[cid]   = doc

    for rank, doc in enumerate(sparse_results, start=1):
        cid = doc["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        meta[cid]   = doc

    # 점수 내림차순 정렬 후 상위 top_n 반환
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        {**meta[cid], "rrf_score": round(score, 6)}
        for cid, score in ranked
        if score >= min_score
    ]




# ────────────────────────────────────────────────────────────
# 1. 청크 샘플링
# ────────────────────────────────────────────────────────────

def load_all_chunks() -> list[dict]:
    """data/chunks/ 아래 모든 JSONL 파일에서 청크를 로드."""
    chunks = []
    for path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


def sample_chunks(chunks: list[dict], n: int, seed: int) -> list[dict]:
    """청크에서 n개 랜덤 샘플링. 너무 짧은 청크(<100자)는 제외."""
    filtered = [c for c in chunks if c.get("char_count", 0) >= 100]
    random.seed(seed)
    return random.sample(filtered, n)


# ────────────────────────────────────────────────────────────
# 2. LLM으로 질문 자동 생성
# ────────────────────────────────────────────────────────────

def generate_question(chunk: dict) -> str | None:
    """qwen3:8b로 청크 본문에서 질문 1개 생성."""
    prompt = f"""다음은 한국 선급 규정의 일부입니다.
이 내용을 바탕으로 실제 실무자가 검색할 법한 자연스러운 질문을 한국어로 1개만 만들어주세요.

규칙:
- "다음 중", "무엇인가요?" 같은 객관식 형태 금지
- 규정 내용에서 직접 답을 찾을 수 있는 질문
- 짧고 명확하게

규정 내용:
{chunk['text'][:400]}

질문:"""

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": QWEN_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 50,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        question = resp.json()["response"].strip()
        # "/think" 태그 제거 (qwen3 thinking 모드 출력 방지)
        if "</think>" in question:
            question = question.split("</think>")[-1].strip()
        return question if question else None
    except Exception as e:
        print(f"  [경고] 질문 생성 실패: {e}")
        return None


def build_eval_set(sampled: list[dict]) -> list[dict]:
    """샘플 청크로 정답셋 생성 후 eval_set.jsonl 저장."""
    if EVAL_SET_PATH.exists():
        print(f"  기존 eval_set.jsonl 발견 → 재사용")
        with open(EVAL_SET_PATH, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    print(f"  질문 생성 중... ({len(sampled)}개)")
    eval_set = []
    for i, chunk in enumerate(sampled, 1):
        question = generate_question(chunk)
        if question:
            item = {
                "question":  question,
                "chunk_id":  chunk["chunk_id"],
                "source":    chunk["source"],
            }
            eval_set.append(item)
            print(f"  [{i:02d}/{len(sampled)}] {chunk['chunk_id']}")
            print(f"         Q: {question}")

    EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_SET_PATH, "w", encoding="utf-8") as f:
        for item in eval_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  → eval_set.jsonl 저장 완료 ({len(eval_set)}개)")
    return eval_set


# ────────────────────────────────────────────────────────────
# 3. Recall@k 계산
# ────────────────────────────────────────────────────────────

def recall_at_k(
    eval_set: list[dict],
    client: QdrantClient,
    sparse_model: SparseTextEmbedding,
    k_list: list[int],
) -> dict[int, float]:
    """각 질문으로 검색 후 Recall@k 계산."""
    hits = {k: 0 for k in k_list}
    total = len(eval_set)

    for i, item in enumerate(eval_set, 1):
        question   = item["question"]
        correct_id = item["chunk_id"]

        dense_vec      = embed_dense(question)
        sparse_vec     = embed_sparse(question, sparse_model)
        dense_results  = search_dense(client, dense_vec, CANDIDATE_K)
        sparse_results = search_sparse(client, sparse_vec, CANDIDATE_K)
        results        = reciprocal_rank_fusion(dense_results, sparse_results)

        retrieved_ids = [r["chunk_id"] for r in results]

        for k in k_list:
            if correct_id in retrieved_ids[:k]:
                hits[k] += 1

        print(f"  [{i:02d}/{total}] {'✅' if correct_id in retrieved_ids[:max(k_list)] else '❌'} {question[:40]}")

    return {k: round(hits[k] / total, 4) for k in k_list}


# ────────────────────────────────────────────────────────────
# 4. main
# ────────────────────────────────────────────────────────────

def main():
    print("=== 04_evaluate.py 시작 ===\n")

    # 초기화
    client       = QdrantClient(url=QDRANT_URL)
    sparse_model = SparseTextEmbedding(model_name=BM25_MODEL_NAME)

    # 1. 청크 샘플링
    print("[1단계] 청크 로드 및 샘플링")
    all_chunks = load_all_chunks()
    print(f"  전체 청크: {len(all_chunks)}개")
    sampled = sample_chunks(all_chunks, SAMPLE_SIZE, RANDOM_SEED)
    print(f"  샘플링: {len(sampled)}개\n")

    # 2. 정답셋 생성
    print("[2단계] 정답셋 생성 (qwen3:8b)")
    eval_set = build_eval_set(sampled)
    print()

    # 3. Recall@k 평가
    print("[3단계] Recall@k 평가")
    t0      = time.time()
    recalls = recall_at_k(eval_set, client, sparse_model, EVAL_K_LIST)
    elapsed = time.time() - t0

    # 4. 결과 출력
    print(f"\n{'='*40}")
    print("📊 평가 결과")
    print(f"{'='*40}")
    for k, score in recalls.items():
        bar = "█" * int(score * 20)
        print(f"  Recall@{k} : {score:.4f}  {bar}")
    print(f"\n  평가 질문 수 : {len(eval_set)}개")
    print(f"  소요 시간   : {elapsed:.1f}s")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()