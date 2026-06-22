# -*- coding: utf-8 -*-
"""
[2~4단계] 임베딩(BGE-M3) -> Qdrant 적재 -> 검색
컨테이너 테스트에서는 TF-IDF로 흉내냈지만, 이 파일은 실제 BGE-M3 임베딩
모델과 Docker로 띄운 Qdrant 서버를 사용하는 정식 버전이다.

[사전 준비 - VM에서 실행 전에 꼭 확인]
1. Qdrant를 Docker로 먼저 띄워야 한다.
   터미널에서:
     docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   (백그라운드로 띄우고 싶으면 -d 옵션 추가: docker run -d -p 6333:6333 ...)

2. 필요한 파이썬 패키지 설치:
     pip install sentence-transformers qdrant-client

3. BGE-M3 모델은 처음 실행할 때 자동으로 Hugging Face에서 다운로드된다
   (약 2GB, 인터넷 연결 필요, 첫 실행만 시간이 좀 걸림. 이후엔 캐시돼서 빠름)

4. 01_chunking.py를 먼저 실행해서 chunks.json이 만들어져 있어야 한다.
"""
import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

CHUNKS_PATH = "./chunks.json"
COLLECTION_NAME = "biz_plan_chunks"

# BGE-M3는 항상 1024차원 벡터를 출력한다 (TF-IDF 테스트 때의 300과 다름)
VECTOR_DIM = 1024

# Qdrant Docker 서버 접속 정보 (위 안내대로 docker run 했다면 기본값 그대로 사용)
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


def build_pipeline():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    texts = [c["text"] for c in chunks]

    # ---- 2단계: 임베딩 (실제 BGE-M3 모델 사용) ----
    print("BGE-M3 모델 로드 중... (처음 실행 시 다운로드로 시간이 걸릴 수 있음)")
    model = SentenceTransformer("BAAI/bge-m3")
    vectors = model.encode(texts, show_progress_bar=True)
    print(f"[임베딩] {len(vectors)}개 청크 -> 벡터 차원 {vectors.shape[1]}")

    # ---- 3단계: Qdrant 적재 (Docker 서버에 연결) ----
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=c["id"],
            vector=vectors[i].tolist(),
            payload={
                "source_file": c["source_file"],
                "item_title": c["item_title"],
                "section": c["section"],
                "text": c["text"],
            },
        )
        for i, c in enumerate(chunks)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[Qdrant 적재] {len(points)}개 포인트 저장 완료 (Docker 서버: {QDRANT_HOST}:{QDRANT_PORT})\n")

    return client, model


def search(client, model, query: str, top_k: int = 3):
    query_vec = model.encode([query])[0].tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
    ).points

    print(f"질문: \"{query}\"")
    print("-" * 60)
    for r in results:
        p = r.payload
        preview = p["text"][:70].replace("\n", " ")
        print(f"  유사도 {r.score:.3f} | [{p['item_title']}] ({p['section']})")
        print(f"    -> {preview}...")
    print()


def main():
    client, model = build_pipeline()

    # 실제 양식 항목에 맞춘 검색 쿼리 예시 (필요에 맞게 수정해서 사용)
    test_queries = [
        "당뇨병 환자를 위한 건강 관리 앱의 문제인식을 작성해줘",
        "반려동물 건강 모니터링 디바이스의 차별화 방안은?",
        "식품 폐기물을 활용한 친환경 사업의 목표시장은?",
        "초등학생 학습 콘텐츠 추천 서비스의 팀 구성",
        "X-Ray 의료 영상 소프트웨어의 사업화 전략",
    ]

    print("=" * 60)
    print("검색 테스트: 업종별 쿼리에 맞는 청크가 검색되는지 확인")
    print("=" * 60 + "\n")

    for q in test_queries:
        search(client, model, q)


if __name__ == "__main__":
    main()
