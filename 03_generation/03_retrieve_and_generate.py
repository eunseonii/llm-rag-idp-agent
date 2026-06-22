# -*- coding: utf-8 -*-
"""
[5단계] 검색(Retrieval) + 생성(Generation) 연결
앞서 검증한 02_embedding_qdrant_search.py의 검색 결과를 받아서,
Ollama로 띄운 Qwen3에 프롬프트로 전달해 실제 양식 항목 텍스트를 생성한다.

[사전 준비 - VM에서 실행 전에 꼭 확인]
1. Qdrant가 Docker로 떠 있어야 한다 (이전 단계와 동일)
     docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

2. Ollama가 설치되어 있고 Qwen3 모델을 받아둔 상태여야 한다.
     # Ollama 설치 (안 되어 있다면)
     curl -fsSL https://ollama.com/install.sh | sh
     # Qwen3 모델 받기 (한 번만, 용량에 따라 시간 소요)
     ollama pull qwen3:8b
     # Ollama 서버 실행 (보통 설치 시 자동으로 백그라운드 서비스로 등록됨)
     ollama serve   # 이미 서비스로 떠 있다면 이 명령은 필요 없음

3. requests 패키지가 필요하다.
     pip install requests

4. 이전 단계(02_embedding_qdrant_search.py)를 한 번 실행해서
   Qdrant 컬렉션에 청크가 이미 적재되어 있어야 한다.
   (이 스크립트는 검색만 다시 수행하고, 적재는 새로 하지 않는다)
"""
import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION_NAME = "biz_plan_chunks"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"  # 받아둔 모델명에 맞게 수정 (예: qwen2.5:7b 등)

TOP_K = 3  # 검색해서 가져올 청크 개수


def search_chunks(client, model, query: str, top_k: int = TOP_K):
    """1~4단계에서 검증한 검색 로직 (그대로 재사용)"""
    query_vec = model.encode([query])[0].tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
    ).points
    return [r.payload for r in results]


def build_prompt(section_name: str, user_concept: str, retrieved_chunks: list):
    """검색된 청크들을 모아 LLM에게 줄 프롬프트를 구성한다"""
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[참고자료 {i} - {chunk['item_title']} / {chunk['section']}]\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_blocks)

    prompt = f"""당신은 정부 창업지원사업 사업계획서 작성을 돕는 어시스턴트입니다.
아래 참고자료들을 참고하여, 사용자가 입력한 아이템 컨셉에 맞는
"{section_name}" 항목을 작성해주세요.

[작성 규칙]
- 참고자료의 문체와 구조(◦, - 등 글머리 기호 사용)를 따라 작성할 것
- 참고자료에 없는 사실을 지어내지 말고, 참고자료의 논리 구조만 빌려서
  사용자의 아이템에 맞게 새로 작성할 것
- 구체적 숫자가 필요한 부분은 OO, ○○ 등으로 표시할 것

[사용자가 입력한 아이템 컨셉]
{user_concept}

[참고자료]
{context_text}

[작성할 항목]
{section_name}
"""
    return prompt


def call_ollama(prompt: str, model: str = OLLAMA_MODEL):
    """Ollama API에 프롬프트를 보내고 생성된 텍스트를 받는다"""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def main():
    # 검색에 쓸 임베딩 모델과 Qdrant 클라이언트 준비 (이전 단계와 동일)
    print("BGE-M3 모델 로드 중...")
    model = SentenceTransformer("BAAI/bge-m3")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # ===== 여기를 원하는 항목/컨셉으로 바꿔서 테스트 =====
    section_name = "1. 문제인식"
    user_concept = "반려동물 비만 관리를 위한 AI 식단·운동 코칭 앱"
    # ====================================================

    print(f"\n[검색] 컨셉 \"{user_concept}\" 관련 청크 검색 중...")
    retrieved = search_chunks(client, model, user_concept, top_k=TOP_K)

    print(f"[검색 결과] {len(retrieved)}건 검색됨:")
    for r in retrieved:
        print(f"  - [{r['item_title']}] ({r['section']})")

    prompt = build_prompt(section_name, user_concept, retrieved)

    print(f"\n[생성] Ollama({OLLAMA_MODEL})에 프롬프트 전송 중...")
    generated_text = call_ollama(prompt)

    print("\n" + "=" * 60)
    print(f"생성된 \"{section_name}\" 항목")
    print("=" * 60)
    print(generated_text)


if __name__ == "__main__":
    main()
