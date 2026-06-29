"""
06_compare_models.py

Granite-embedding vs BGE-M3 검색 품질 비교.
동일한 질의와 정답셋으로 두 컬렉션의 Recall@k를 측정하여 비교한다.

비교 대상:
  - kr_rules_study : Granite-embedding:278m (Dense 768d)
  - kr_rules_bge   : BGE-M3 (Dense 1024d)
"""

import json
import time
import requests
from pathlib import Path
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import SparseVector

# ── 설정 ────────────────────────────────────────────────────
QDRANT_URL       = "http://localhost:6333"
OLLAMA_URL       = "http://localhost:11434"
BM25_MODEL_NAME  = "Qdrant/bm25"
EVAL_SET_PATH    = Path(__file__).parent.parent / "data" / "eval_set.jsonl"

GRANITE_MODEL    = "granite-embedding:278m"
BGE_MODEL        = "bge-m3:latest"

CANDIDATE_K      = 20
TOP_K            = 4
RRF_K            = 60
EVAL_K_LIST      = [1, 2, 4]

COLLECTIONS = {
    "Granite": {
        "name":  "kr_rules_study",
        "model": "granite",
    },
    "BGE-M3": {
        "name":  "kr_rules_bge",
        "model": "bge",
    },
}

SAMPLE_QUERIES = [
    "선박의 복원성 기준은 무엇인가?",
    "화재 탐지 시스템 설치 요건",
    "선급 검사 주기와 종류",
]


# ────────────────────────────────────────────────────────────
# 1. 임베딩
# ────────────────────────────────────────────────────────────

def embed_dense(query: str, model: str) -> list[float]:
    """Ollama로 Dense 벡터 생성. model: 'granite' | 'bge'"""
    model_name = GRANITE_MODEL if model == "granite" else BGE_MODEL
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": model_name, "prompt": query},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def embed_sparse(query: str, sparse_model: SparseTextEmbedding) -> SparseVector:
    """fastembed BM25로 Sparse 벡터 생성."""
    result = list(sparse_model.embed([query]))[0]
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


# ────────────────────────────────────────────────────────────
# 2. 검색
# ────────────────────────────────────────────────────────────

def search_dense(client: QdrantClient, collection: str, vector: list[float]) -> list[dict]:
    response = client.query_points(
        collection_name=collection,
        query=vector,
        using="dense",
        limit=CANDIDATE_K,
        with_payload=True,
    )
    return [
        {
            "chunk_id": p.payload.get("chunk_id", ""),
            "score":    p.score,
            "text":     p.payload.get("text", ""),
        }
        for p in response.points
    ]


def search_sparse(client: QdrantClient, collection: str, vector: SparseVector) -> list[dict]:
    response = client.query_points(
        collection_name=collection,
        query=models.SparseVector(
            indices=vector.indices,
            values=vector.values,
        ),
        using="sparse",
        limit=CANDIDATE_K,
        with_payload=True,
    )
    return [
        {
            "chunk_id": p.payload.get("chunk_id", ""),
            "score":    p.score,
            "text":     p.payload.get("text", ""),
        }
        for p in response.points
    ]


def rrf(dense: list[dict], sparse: list[dict]) -> list[dict]:
    """RRF 융합."""
    scores: dict[str, float] = {}
    meta:   dict[str, dict]  = {}

    for rank, doc in enumerate(dense, 1):
        cid = doc["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        meta[cid]   = doc

    for rank, doc in enumerate(sparse, 1):
        cid = doc["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        meta[cid]   = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    return [{**meta[cid], "rrf_score": round(score, 6)} for cid, score in ranked]


def hybrid_search(
    query: str,
    collection: str,
    model: str,
    client: QdrantClient,
    sparse_model: SparseTextEmbedding,
) -> list[dict]:
    dense_vec      = embed_dense(query, model)
    sparse_vec     = embed_sparse(query, sparse_model)
    dense_results  = search_dense(client, collection, dense_vec)
    sparse_results = search_sparse(client, collection, sparse_vec)
    return rrf(dense_results, sparse_results)


# ────────────────────────────────────────────────────────────
# 3. 질의 결과 비교 출력
# ────────────────────────────────────────────────────────────

def compare_queries(client: QdrantClient, sparse_model: SparseTextEmbedding) -> None:
    """샘플 질의로 두 컬렉션 결과 나란히 출력."""
    for query in SAMPLE_QUERIES:
        print(f"\n{'='*70}")
        print(f"질의: {query}")
        print(f"{'='*70}")
        print(f"\n{'[Granite]':<35} {'[BGE-M3]'}")
        print("-" * 70)

        g_results = hybrid_search(
            query, "kr_rules_study", "granite", client, sparse_model
        )
        b_results = hybrid_search(
            query, "kr_rules_bge", "bge", client, sparse_model
        )

        for i in range(TOP_K):
            g = g_results[i]["chunk_id"] if i < len(g_results) else "-"
            b = b_results[i]["chunk_id"] if i < len(b_results) else "-"
            match = "✅" if g == b else "  "
            print(f"{i+1}. {g:<33} {b}  {match}")

        # 겹치는 문서
        g_ids   = {r["chunk_id"] for r in g_results}
        b_ids   = {r["chunk_id"] for r in b_results}
        overlap = g_ids & b_ids
        print(f"\n  → 공통 문서: {len(overlap)}개 / {TOP_K}개")


# ────────────────────────────────────────────────────────────
# 4. Recall@k 비교
# ────────────────────────────────────────────────────────────

def compare_recall(client: QdrantClient, sparse_model: SparseTextEmbedding) -> None:
    """eval_set.jsonl로 두 컬렉션의 Recall@k 비교."""
    if not EVAL_SET_PATH.exists():
        print("  eval_set.jsonl 없음 → 04_evaluate.py 먼저 실행하세요")
        return

    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = [json.loads(l) for l in f if l.strip()]

    print(f"\n  평가 질문 수: {len(eval_set)}개\n")

    results = {}
    for label, cfg in COLLECTIONS.items():
        hits    = {k: 0 for k in EVAL_K_LIST}
        total   = len(eval_set)

        for item in eval_set:
            query      = item["question"]
            correct_id = item["chunk_id"]

            search_results = hybrid_search(
                query, cfg["name"], cfg["model"], client, sparse_model
            )
            retrieved = [r["chunk_id"] for r in search_results]

            for k in EVAL_K_LIST:
                if correct_id in retrieved[:k]:
                    hits[k] += 1

        results[label] = {k: round(hits[k] / total, 4) for k in EVAL_K_LIST}
        print(f"  [{label}] 완료")

    # 결과 출력
    print(f"\n{'='*50}")
    print("📊 Recall@k 비교")
    print(f"{'='*50}")
    print(f"{'':10} {'Granite':>10} {'BGE-M3':>10} {'승자':>8}")
    print("-" * 50)

    for k in EVAL_K_LIST:
        g = results["Granite"][k]
        b = results["BGE-M3"][k]
        winner = "Granite ✅" if g > b else "BGE-M3 ✅" if b > g else "동점"
        print(f"Recall@{k}  {g:>10.4f} {b:>10.4f} {winner:>8}")


# ────────────────────────────────────────────────────────────
# 5. main
# ────────────────────────────────────────────────────────────

def main():
    print("=== 06_compare_models.py 시작 ===\n")

    client       = QdrantClient(url=QDRANT_URL)
    sparse_model = SparseTextEmbedding(model_name=BM25_MODEL_NAME)

    # ── 모드 선택 ──────────────────────
    MODE = "both"   # "query" | "recall" | "both"
    # ───────────────────────────────────

    if MODE in ("query", "both"):
        print("[1단계] 샘플 질의 비교")
        compare_queries(client, sparse_model)

    if MODE in ("recall", "both"):
        print("\n[2단계] Recall@k 비교")
        compare_recall(client, sparse_model)

    print(f"\n{'='*50}")
    print("완료!")


if __name__ == "__main__":
    main()