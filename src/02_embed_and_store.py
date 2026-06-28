"""
02_embed_and_store.py
BGE-M3 Dense + Sparse 임베딩 → Qdrant 저장

[의도적 버그 포함]
PUA 문자 방어 코드를 넣지 않아서 400 에러가 재현됩니다.
디버깅 후 fix_pua() 함수를 활성화해서 해결하세요.

실행:
    python src/02_embed_and_store.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Generator

# FlagEmbedding: BGE-M3 Dense + Sparse 동시 추출 라이브러리
# pip install FlagEmbedding
from FlagEmbedding import BGEM3FlagModel

# Qdrant 클라이언트
# pip install qdrant-client
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

# ── 설정값 ──────────────────────────────────────────────────────────────────
CHUNKS_DIR = Path("data/chunks")          # 01에서 만든 JSONL들
COLLECTION  = "kr_rules_study"            # Qdrant 컬렉션 이름
QDRANT_URL  = "http://localhost:6333"     # 도커로 뜬 Qdrant
BATCH_SIZE  = 32                          # 한 번에 임베딩할 청크 수
                                          # (GPU 없으면 16~32가 안정적)
DENSE_DIM   = 1024                        # BGE-M3 Dense 벡터 차원
MODEL_NAME  = "BAAI/bge-m3"              # HuggingFace 모델 ID

LOG_FILE    = Path("logs/02_embed_and_store.log")

# Qdrant ID는 정수만 허용 → 전체 청크에 전역 순번 부여
_point_counter = 0
# ────────────────────────────────────────────────────────────────────────────

# ── 로거 설정 ────────────────────────────────────────────────────────────────
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)
# ────────────────────────────────────────────────────────────────────────────


# ── [버그 재현용] PUA 제거 함수 — 지금은 비활성화 상태 ──────────────────────
# PUA(Private Use Area, U+E000~U+F8FF): PDF에서 폰트 매핑 오류로 생기는
# 의미없는 유니코드 문자들. Qdrant에 JSON으로 전송할 때 400 에러를 유발한다.
#
# 버그 재현 후 아래 함수를 활성화하고, clean() 호출부의 주석을 해제하면 fix됩니다.

import re
_PUA_RE = re.compile(r"[\uE000-\uF8FF]")

def fix_pua(text: str) -> str:
    """PUA 문자를 빈 문자열로 치환한다."""
    return _PUA_RE.sub("", text)


def load_chunks(chunks_dir: Path) -> Generator[dict, None, None]:
    """
    data/chunks/*.jsonl 을 하나씩 읽어서 dict를 yield한다.

    왜 Generator?
    - 6,732개 청크를 한꺼번에 메모리에 올리면 ~수백MB 차지
    - yield로 배치 단위로 흘려보내면 메모리 효율적
    """
    jsonl_files = sorted(chunks_dir.glob("*.jsonl"))
    if not jsonl_files:
        log.error(f"JSONL 파일 없음: {chunks_dir}")
        sys.exit(1)

    for fpath in jsonl_files:
        log.info(f"로드 중: {fpath.name}")
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def batched(iterable, n: int) -> Generator[list, None, None]:
    """
    이터러블을 n개짜리 리스트로 잘라서 yield한다.
    예) [1,2,3,4,5], n=2 → [1,2], [3,4], [5]
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def setup_collection(client: QdrantClient) -> None:
    """
    Qdrant에 kr_rules_study 컬렉션을 만든다.
    이미 있으면 삭제 후 재생성 (개발 중 반복 실행 편의를 위해).

    컬렉션 구조:
    - dense  : 1024차원 float 벡터, Cosine 유사도
    - sparse : sparse 벡터 (인덱스+값 쌍), 내적 유사도
                → BM25와 비슷하게 키워드 매칭에 강함
    """
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        log.warning(f"기존 컬렉션 삭제: {COLLECTION}")
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
            # SparseVectorParams에는 차원 지정이 없음
            # sparse 벡터는 실제로 등장한 토큰 인덱스만 저장하기 때문
        },
    )
    log.info(f"컬렉션 생성 완료: {COLLECTION}")


def embed_batch(model: BGEM3FlagModel, texts: list[str]) -> dict:
    """
    BGE-M3로 Dense + Sparse 벡터를 동시에 추출한다.

    반환값 구조:
    {
      "dense_vecs":  [[float, ...], ...],   # (batch, 1024)
      "lexical_weights": [{token_id: weight, ...}, ...],  # sparse
    }

    왜 BGE-M3?
    - Dense만 있는 모델과 달리 Sparse(어휘 가중치)도 뽑아준다
    - 한 번 인퍼런스로 두 벡터를 얻으니 속도 이점
    - 한국어 포함 다국어 지원
    """
    output = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        max_length=512,          # 512 토큰 초과분은 잘림
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,  # ColBERT는 이번 과제에서 안 씀
    )
    return output


def make_point(chunk: dict, dense_vec: list[float], sparse_weights: dict) -> PointStruct:
    """
    Qdrant에 넣을 PointStruct 하나를 만든다.

    PointStruct 구성:
    - id      : chunk_id (정수)
    - vector  : {"dense": [...], "sparse": SparseVector}
    - payload : 검색 결과와 함께 반환할 메타데이터

    SparseVector는 (indices, values) 쌍으로 구성:
    - indices: 어떤 토큰이 등장했는가 (int 리스트)
    - values : 그 토큰의 가중치 (float 리스트)
    """
    global _point_counter
    text = chunk["text"]
    # ── [버그 재현 포인트] ──────────────────────────────────────────────────
    # PUA 문자가 있는 text를 그대로 payload에 넣으면
    # Qdrant가 JSON 파싱 실패 → HTTP 400 반환
    #
    # 수정하려면 아래 주석을 해제:
    text = fix_pua(text)
    # ───────────────────────────────────────────────────────────────────────

    # sparse_weights: {token_id(str): weight(float)}
    # Qdrant SparseVector는 indices를 int로 받아야 한다
    indices = [int(k) for k in sparse_weights.keys()]
    values  = [float(v) for v in sparse_weights.values()]

    point_id = _point_counter
    _point_counter += 1

    return PointStruct(
        id=point_id,
        vector={
            "dense":  dense_vec,
            "sparse": SparseVector(indices=indices, values=values),
        },
        payload={
            "text":      text,
            "source":    chunk.get("source", ""),
            "chunk_id":  chunk.get("chunk_id", ""),
            "page":      chunk.get("page", -1),
            "chunk_idx": chunk.get("chunk_index", -1),
        },
    )


def main() -> None:
    log.info("=== 02_embed_and_store 시작 ===")

    # 1) BGE-M3 모델 로드
    #    use_fp16=True: float16으로 올려서 메모리 절반으로 줄임
    #    CPU라면 False로 바꾸거나 그냥 두면 됨 (fp16은 GPU에서 효과적)
    log.info(f"모델 로드: {MODEL_NAME}")
    model = BGEM3FlagModel(MODEL_NAME, use_fp16=False)
    log.info("모델 로드 완료")

    # 2) Qdrant 연결
    client = QdrantClient(url=QDRANT_URL)
    log.info(f"Qdrant 연결: {QDRANT_URL}")

    # 3) 컬렉션 준비
    setup_collection(client)

    # 4) 청크 로드 → 배치 임베딩 → Qdrant upsert
    chunks_gen = load_chunks(CHUNKS_DIR)
    total_points = 0
    error_count  = 0

    for batch_idx, batch in enumerate(batched(chunks_gen, BATCH_SIZE)):
        texts = [c["text"] for c in batch]

        # 임베딩
        t0 = time.time()
        try:
            output = embed_batch(model, texts)
        except Exception as e:
            log.error(f"임베딩 실패 (배치 {batch_idx}): {e}")
            error_count += 1
            continue
        elapsed_embed = time.time() - t0

        # PointStruct 구성
        points = []
        for i, chunk in enumerate(batch):
            dense_vec      = output["dense_vecs"][i].tolist()
            sparse_weights = output["lexical_weights"][i]  # dict
            points.append(make_point(chunk, dense_vec, sparse_weights))

        # Qdrant upsert
        #   upsert = insert + update (id가 같으면 덮어씀)
        t1 = time.time()
        try:
            client.upsert(collection_name=COLLECTION, points=points)
            elapsed_upsert = time.time() - t1
            total_points += len(points)
            log.info(
                f"배치 {batch_idx:04d} | {len(points)}건 | "
                f"embed {elapsed_embed:.1f}s | upsert {elapsed_upsert:.2f}s | "
                f"누계 {total_points}"
            )
        except Exception as e:
            # ── 여기서 PUA 400 에러가 터집니다 ──────────────────────────────
            log.error(f"Qdrant upsert 실패 (배치 {batch_idx}): {e}")
            log.error(f"  문제 청크 샘플: {texts[0][:120]!r}")
            error_count += 1
            # continue로 넘기면 나머지 배치는 계속 시도
            continue

    log.info(f"=== 완료 | 저장 {total_points}건 | 에러 배치 {error_count}개 ===")

    # 5) 저장 결과 확인
    info = client.get_collection(COLLECTION)
    log.info(f"컬렉션 포인트 수: {info.points_count}")


if __name__ == "__main__":
    main()
