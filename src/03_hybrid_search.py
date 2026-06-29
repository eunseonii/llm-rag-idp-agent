"""
03_hybrid_search.py

Dense(Granite-embedding) + Sparse(BM25) 하이브리드 검색 모듈.
Qdrant의 query_points API를 사용해 두 검색 결과를 받고,
RRF(Reciprocal Rank Fusion)로 직접 재정렬하여 상위 k개를 반환한다.

파이프라인 위치: 02_embed_and_store.py → [현재] → 04_evaluate.py
"""

import time
import requests
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import QdrantClient, models
from qdrant_client.models import SparseVector

# ── 설정 ────────────────────────────────────────────────────
QDRANT_URL       = "http://localhost:6333"
COLLECTION_NAME  = "kr_rules_study"
OLLAMA_URL       = "http://localhost:11434"
GRANITE_MODEL    = "granite-embedding:278m"
BM25_MODEL_NAME  = "Qdrant/bm25"

TOP_K            = 4    # 최종 반환 개수
CANDIDATE_K      = 20   # 각 검색기에서 뽑을 후보 개수
RRF_K            = 60   # RRF 상수 (논문 기본값)

# ── 샘플 질의 ────────────────────────────────────────────────
SAMPLE_QUERIES = [
    "선박의 복원성 기준은 무엇인가?",
    "화재 탐지 시스템 설치 요건",
    "선급 검사 주기와 종류",
]


# ────────────────────────────────────────────────────────────
# 1. 임베딩 함수
# ────────────────────────────────────────────────────────────

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
    ]


# ────────────────────────────────────────────────────────────
# 4. 하이브리드 검색 통합 함수
# ────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    client: QdrantClient,
    sparse_model: SparseTextEmbedding,
) -> list[dict]:
    """Dense + Sparse → RRF → 상위 TOP_K 반환."""
    dense_vec  = embed_dense(query)
    sparse_vec = embed_sparse(query, sparse_model)

    dense_results  = search_dense(client, dense_vec, CANDIDATE_K)
    sparse_results = search_sparse(client, sparse_vec, CANDIDATE_K)

    return reciprocal_rank_fusion(dense_results, sparse_results)


# ────────────────────────────────────────────────────────────
# 5. 결과 출력 헬퍼
# ────────────────────────────────────────────────────────────

def print_results(query: str, results: list[dict]) -> None:
    print(f"\n{'='*60}")
    print(f"질의: {query}")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}위] rrf_score={r['rrf_score']}")
        print(f"  chunk_id : {r['chunk_id']}")
        print(f"  source   : {r['source']}  (page {r['page']})")
        print(f"  본문 미리보기: {r['text'][:120].strip()}...")


# ────────────────────────────────────────────────────────────
# 6. main
# ────────────────────────────────────────────────────────────

def compare_search_modes(
    query: str,
    client: QdrantClient,
    sparse_model: SparseTextEmbedding,
) -> None:
    """Dense / Sparse / Hybrid 결과를 나란히 출력해서 비교."""
    dense_vec  = embed_dense(query)
    sparse_vec = embed_sparse(query, sparse_model)

    dense_results  = search_dense(client, dense_vec, TOP_K)
    sparse_results = search_sparse(client, sparse_vec, TOP_K)
    hybrid_results = reciprocal_rank_fusion(dense_results, sparse_results)

    print(f"\n{'='*70}")
    print(f"질의: {query}")
    print(f"{'='*70}")

    # 헤더
    print(f"\n{'[Dense]':<35} {'[Sparse]':<35} {'[Hybrid/RRF]'}")
    print("-" * 105)

    for i in range(TOP_K):
        d = dense_results[i]["chunk_id"]  if i < len(dense_results)  else "-"
        s = sparse_results[i]["chunk_id"] if i < len(sparse_results) else "-"
        h = hybrid_results[i]["chunk_id"] if i < len(hybrid_results) else "-"
        print(f"{i+1}. {d:<33} {s:<33} {h}")

    # 겹치는 문서 표시
    dense_ids  = {r["chunk_id"] for r in dense_results}
    sparse_ids = {r["chunk_id"] for r in sparse_results}
    overlap    = dense_ids & sparse_ids
    print(f"\n  → Dense∩Sparse 겹치는 문서: {len(overlap)}개 / {TOP_K}개")
    if overlap:
        for cid in overlap:
            print(f"     - {cid}")


def main():
    print("=== 03_hybrid_search.py 시작 ===\n")

    client       = QdrantClient(url=QDRANT_URL)
    sparse_model = SparseTextEmbedding(model_name=BM25_MODEL_NAME)
    print(f"Qdrant 연결: {QDRANT_URL}")
    print(f"컬렉션: {COLLECTION_NAME}\n")

    # ── 모드 선택 ──────────────────────────────
    MODE = "search"   # "search" | "compare" | "rrf_k"
    # ───────────────────────────────────────────

    if MODE == "search":
        for query in SAMPLE_QUERIES:
            t0      = time.time()
            results = hybrid_search(query, client, sparse_model)
            elapsed = time.time() - t0
            print_results(query, results)
            print(f"\n  검색 소요시간: {elapsed:.2f}s")

    elif MODE == "compare":
        for query in SAMPLE_QUERIES:
            compare_search_modes(query, client, sparse_model)

    elif MODE == "rrf_k":
        query = "선박의 복원성 기준은 무엇인가?"
        dense_vec  = embed_dense(query)
        sparse_vec = embed_sparse(query, sparse_model)
        dense_results  = search_dense(client, dense_vec, CANDIDATE_K)
        sparse_results = search_sparse(client, sparse_vec, CANDIDATE_K)

        print(f"\n질의: {query}")
        print(f"{'='*70}")

        for k in [10, 30, 60]:
            results = reciprocal_rank_fusion(dense_results, sparse_results, k=k)
            print(f"\n  [k={k}]")
            for i, r in enumerate(results, 1):
                print(f"  {i}위 (rrf={r['rrf_score']}) {r['chunk_id']}")

    print(f"\n{'='*70}")
    print("완료!")


if __name__ == "__main__":
    main()