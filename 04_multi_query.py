# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION_NAME = "biz_plan_chunks"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

SECTION_QUERY_TEMPLATES = {
    "1. 문제인식": [
        "{concept} 시장 현황 및 문제점",
        "{concept} 개발 배경 및 필요성",
        "{concept} 목표시장 고객 분석",
        "{concept} 경쟁사 현황",
    ],
    "2. 실현가능성": [
        "{concept} 개발 계획 및 준비 현황",
        "{concept} 차별화 방안 기술력",
        "{concept} 특허 핵심 기능 솔루션",
        "{concept} 구현 방법 산출물",
    ],
    "3. 성장전략": [
        "{concept} 사업화 전략 수익 모델",
        "{concept} 비즈니스 모델 BM",
        "{concept} 시장 진입 전략 마케팅",
        "{concept} 투자유치 자금 조달 로드맵",
    ],
    "4. 팀 구성": [
        "{concept} 대표자 역량 팀 구성",
        "{concept} 외부 협력기관 파트너",
        "{concept} 팀원 보유 역량 경력",
    ],
    "아이템 개요": [
        "{concept} 서비스 개요 핵심 기능",
        "{concept} 제품 특징 고객 혜택",
    ],
}


def multi_query_search(
    client,
    model,
    section: str,
    concept: str,
    top_k_per_query: int = 3,
    final_top_k: int = 5,
):
    templates = SECTION_QUERY_TEMPLATES.get(
        section,
        ["{concept} " + section]
    )

    all_results = {}

    for template in templates:
        query = template.format(concept=concept)
        query_vec = model.encode([query])[0].tolist()

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k_per_query,
        ).points

        for r in results:
            chunk_id = r.id
            if chunk_id not in all_results or r.score > all_results[chunk_id][1]:
                all_results[chunk_id] = (r.payload, r.score)

    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x[1],
        reverse=True
    )[:final_top_k]

    return sorted_results


def main():
    print("BGE-M3 모델 로드 중...")
    model = SentenceTransformer("BAAI/bge-m3")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    test_cases = [
        ("1. 문제인식", "당뇨병 환자를 위한 건강 관리 앱"),
        ("3. 성장전략", "반려동물 비만 관리 IoT 디바이스"),
        ("2. 실현가능성", "초등학생 AI 학습 콘텐츠 추천 플랫폼"),
        ("4. 팀 구성", "친환경 대체식품 소재 스타트업"),
    ]

    for section, concept in test_cases:
        print("\n" + "=" * 60)
        print(f"항목: [{section}] | 컨셉: \"{concept}\"")
        print("=" * 60)

        results = multi_query_search(
            client, model, section, concept,
            top_k_per_query=3,
            final_top_k=5,
        )

        print(f"검색된 청크 {len(results)}개:")
        for payload, score in results:
            preview = payload["text"][:60].replace("\n", " ")
            print(f"  {score:.3f} | [{payload['item_title'][:30]}] "
                  f"({payload['section']})")
            print(f"         -> {preview}...")


if __name__ == "__main__":
    main()