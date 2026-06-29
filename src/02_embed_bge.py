"""
02_embed_bge.py

BGE-M3(Ollama) + fastembed BM25 임베딩 후 Qdrant 저장.
Granite 버전(02_embed_and_store.py)과 비교용으로 별도 컬렉션에 저장.

컬렉션명: kr_rules_bge (kr_rules_study와 독립)
파이프라인 위치: 01_parse_and_chunk.py → [현재] → 03_hybrid_search.py
"""

import json
import time
import requests
from pathlib import Path
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
)

# ── 설정 ────────────────────────────────────────────────────
CHUNKS_DIR      = Path(__file__).parent.parent / "data" / "chunks"
QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "kr_rules_bge"        # Granite 버전과 분리
OLLAMA_URL      = "http://localhost:11434"
BGE_MODEL       = "bge-m3:latest"
BM25_MODEL_NAME = "Qdrant/bm25"

DENSE_DIM       = 1024   # BGE-M3 출력 차원
BATCH_SIZE      = 5      # BGE-M3는 느리므로 작게
LOG_INTERVAL    = 50     # 몇 개마다 진행상황 출력


# ────────────────────────────────────────────────────────────
# 1. Qdrant 컬렉션 초기화
# ────────────────────────────────────────────────────────────

def init_collection(client: QdrantClient) -> None:
    """kr_rules_bge 컬렉션 생성 (이미 있으면 삭제 후 재생성)."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"  기존 컬렉션 '{COLLECTION_NAME}' 삭제 후 재생성")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    print(f"  컬렉션 '{COLLECTION_NAME}' 생성 완료 (dense={DENSE_DIM}차원)")


# ────────────────────────────────────────────────────────────
# 2. 임베딩
# ────────────────────────────────────────────────────────────

def embed_dense_bge(text: str) -> list[float]:
    """Ollama BGE-M3로 Dense 벡터 생성."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": BGE_MODEL, "prompt": text},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def embed_sparse_bm25(text: str, model: SparseTextEmbedding) -> dict:
    """fastembed BM25로 Sparse 벡터 생성."""
    result = list(model.embed([text]))[0]
    return {
        "indices": result.indices.tolist(),
        "values":  result.values.tolist(),
    }


# ────────────────────────────────────────────────────────────
# 3. 청크 로드
# ────────────────────────────────────────────────────────────

def load_chunks() -> list[dict]:
    """data/chunks/ 아래 모든 JSONL 파일에서 청크 로드."""
    chunks = []
    for path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    print(f"  총 {len(chunks)}개 청크 로드")
    return chunks


# ────────────────────────────────────────────────────────────
# 4. 임베딩 및 저장
# ────────────────────────────────────────────────────────────

def embed_and_store(
    chunks: list[dict],
    client: QdrantClient,
    sparse_model: SparseTextEmbedding,
) -> None:
    """청크 전체를 임베딩하고 Qdrant에 배치 저장."""
    total   = len(chunks)
    batch   = []
    t_start = time.time()

    for i, chunk in enumerate(chunks, 1):
        text = chunk["text"]

        dense_vec  = embed_dense_bge(text)
        sparse_vec = embed_sparse_bm25(text, sparse_model)

        point = PointStruct(
            id=i,
            vector={
                "dense":  dense_vec,
                "sparse": models.SparseVector(
                    indices=sparse_vec["indices"],
                    values=sparse_vec["values"],
                ),
            },
            payload={
                "text":     text,
                "source":   chunk.get("source", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "page":     chunk.get("page", -1),
            },
        )
        batch.append(point)

        # 배치 저장
        if len(batch) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            batch = []

        # 진행상황 출력
        if i % LOG_INTERVAL == 0 or i == total:
            elapsed = time.time() - t_start
            eta     = elapsed / i * (total - i)
            print(f"  [{i:>5}/{total}] 경과 {elapsed/60:.1f}분 | 남은시간 {eta/60:.1f}분")

    # 남은 배치 저장
    if batch:
        client.upsert(collection_name=COLLECTION_NAME, points=batch)


# ────────────────────────────────────────────────────────────
# 5. main
# ────────────────────────────────────────────────────────────

def main():
    print("=== 02_embed_bge.py 시작 ===\n")
    t0 = time.time()

    client       = QdrantClient(url=QDRANT_URL)
    sparse_model = SparseTextEmbedding(model_name=BM25_MODEL_NAME)

    print("[1단계] 컬렉션 초기화")
    init_collection(client)

    print("\n[2단계] 청크 로드")
    chunks = load_chunks()

    print("\n[3단계] 임베딩 및 저장 (BGE-M3 + BM25)")
    print(f"  배치 크기: {BATCH_SIZE}, 총 {len(chunks)}개")
    embed_and_store(chunks, client, sparse_model)

    total_min = (time.time() - t0) / 60
    print(f"\n총 소요시간: {total_min:.1f}분")
    print("=== 완료 ===")


if __name__ == "__main__":
    main()