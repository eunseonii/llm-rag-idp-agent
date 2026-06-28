"""
02_embed_and_store.py  — v2
Dense(Granite-embedding) + Sparse(fastembed BM25) → Qdrant 저장

[v1 → v2 변경 이유]
- v1: BGE-M3 단일 모델로 Dense+Sparse 동시 처리
  → CPU 환경에서 배치당 70초, 총 3h47m 예상 → 중단
  → 팀 서버에서 NaN 버그 발생 (원인 미해결)
- v2: Granite-embedding(Ollama) + fastembed BM25 분리
  → Ollama API 호출 방식으로 속도 개선 예상
  → NaN 방어 코드 추가 (np.isfinite 체크)

[v1 vs v2 비교 목적]
- 청킹 조건 동일 유지 (PDF 10개, chunk_size=500, overlap=50)
- 추후 04_evaluate.py에서 recall@k로 두 버전 성능 비교 예정

[테스트 방법]
- TEST_MODE = True  → 청크 50개만 처리 (파이프라인 검증용)
- TEST_MODE = False → 전체 6,732개 처리

실행:
    cd ~/kr-rules-rag
    python src/02_embed_and_store.py
"""

# ~/kr-rules-rag/src/02_embed_and_store.py v2

import json
import time
import logging
from pathlib import Path

import requests
import numpy as np
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
)

# ── 로깅 설정 ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/02_embed.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── 설정값 ─────────────────────────────────────────────────
CHUNKS_DIR    = Path("data/chunks")
COLLECTION    = "kr_rules_study"
OLLAMA_URL    = "http://localhost:11434/api/embed"
GRANITE_MODEL = "granite-embedding:278m"
DENSE_DIM     = 768          # Granite-embedding 출력 차원
BATCH_SIZE    = 32           # 한 번에 처리할 청크 수
TEST_MODE     = True        # True면 청크 50개만 처리 (테스트용)
TEST_LIMIT    = 50


# ── 1. Granite Dense 임베딩 (Ollama API) ───────────────────
def embed_dense(texts: list[str]) -> list[list[float]]:
    """
    Ollama API를 호출해서 Dense 벡터를 받아옴.
    반환: [[0.12, -0.34, ...], [0.56, 0.78, ...], ...]  (배치 크기 × 768)
    """
    resp = requests.post(
        OLLAMA_URL,
        json={"model": GRANITE_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    embeddings = resp.json()["embeddings"]  # List[List[float]]

    # NaN/Inf 체크 — v1에서 팀 서버 NaN 버그 재현 방지
    for i, vec in enumerate(embeddings):
        arr = np.array(vec)
        if not np.isfinite(arr).all():
            raise ValueError(f"NaN/Inf detected in dense embedding, index={i}")

    return embeddings


# ── 2. fastembed BM25 Sparse 임베딩 ───────────────────────
def init_sparse_model() -> SparseTextEmbedding:
    """
    BM25 모델 초기화. 첫 실행 시 ONNX 모델 자동 다운로드 (~수십 MB).
    """
    log.info("BM25 sparse model 로딩 중...")
    model = SparseTextEmbedding(model_name="Qdrant/bm25")
    log.info("BM25 sparse model 로딩 완료")
    return model


def embed_sparse(model: SparseTextEmbedding, texts: list[str]):
    """
    BM25 sparse 벡터 생성.
    반환: fastembed SparseEmbedding 객체 리스트
          각 객체는 .indices (단어 ID), .values (TF-IDF 점수) 보유
    """
    return list(model.embed(texts))


# ── 3. Qdrant 컬렉션 생성 ──────────────────────────────────
def create_collection(client: QdrantClient):
    """
    Dense + Sparse 벡터를 함께 저장하는 컬렉션 생성.
    이미 존재하면 삭제 후 재생성 (멱등성 보장).
    """
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        log.warning(f"컬렉션 '{COLLECTION}' 이미 존재 → 삭제 후 재생성")
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            # Dense 벡터 설정
            "dense": VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            # Sparse 벡터 설정 (BM25)
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    log.info(f"컬렉션 '{COLLECTION}' 생성 완료 (dense={DENSE_DIM}차원 + sparse BM25)")


# ── 4. 청크 로딩 ───────────────────────────────────────────
def load_chunks() -> list[dict]:
    chunks = []
    for jsonl_path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    log.info(f"총 {len(chunks)}개 청크 로딩 완료")
    return chunks


# ── 5. 배치 저장 ───────────────────────────────────────────
def store_batch(
    client: QdrantClient,
    sparse_model: SparseTextEmbedding,
    chunks: list[dict],
    start_id: int,
):
    """
    청크 리스트를 받아 Dense + Sparse 임베딩 후 Qdrant에 저장.
    start_id: 이 배치의 첫 번째 포인트 ID
    """
    texts = [c["text"] for c in chunks]

    # Dense 임베딩 (Ollama Granite)
    dense_vecs = embed_dense(texts)

    # Sparse 임베딩 (BM25)
    sparse_vecs = embed_sparse(sparse_model, texts)

    # Qdrant PointStruct 조립
    points = []
    for i, (chunk, dvec, svec) in enumerate(zip(chunks, dense_vecs, sparse_vecs)):
        points.append(
            PointStruct(
                id=start_id + i,
                vector={
                    "dense": dvec,
                    "sparse": SparseVector(
                        indices=svec.indices.tolist(),
                        values=svec.values.tolist(),
                    ),
                },
                payload={
                    "text":     chunk["text"],
                    "source":   chunk.get("source", ""),
                    "page":     chunk.get("page", -1),
                    "chunk_id": chunk.get("chunk_id", start_id + i),
                },
            )
        )

    client.upsert(collection_name=COLLECTION, points=points)


# ── 6. 메인 ────────────────────────────────────────────────
def main():
    log.info("=== 02_embed_and_store.py v2 시작 ===")
    t_total = time.time()

    # 클라이언트 초기화
    client = QdrantClient(host="localhost", port=6333)
    sparse_model = init_sparse_model()

    # 컬렉션 생성
    create_collection(client)

    # 청크 로딩
    chunks = load_chunks()
    if TEST_MODE:
        chunks = chunks[:TEST_LIMIT]
        log.info(f"[TEST MODE] {TEST_LIMIT}개만 처리")

    # 배치 처리
    total = len(chunks)
    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        t_batch = time.time()

        store_batch(client, sparse_model, batch, start_id=batch_start)

        elapsed = time.time() - t_batch
        progress = min(batch_start + BATCH_SIZE, total)
        log.info(
            f"[{progress}/{total}] 배치 완료 | "
            f"배치 소요: {elapsed:.1f}s | "
            f"예상 잔여: {elapsed * (total - progress) / BATCH_SIZE:.0f}s"
        )

    # 최종 검증
    count = client.count(collection_name=COLLECTION).count
    log.info(f"=== 완료: Qdrant 저장 포인트 수 = {count} ===")
    log.info(f"총 소요시간: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()