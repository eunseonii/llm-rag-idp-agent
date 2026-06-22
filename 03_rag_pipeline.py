# -*- coding: utf-8 -*-
"""
[3단계] RAG 파이프라인 (LangChain + MultiQueryRetriever + Ollama)
- 기획서 섹션별 멀티쿼리 검색
- Ollama Qwen3:8b로 텍스트 생성
- FastAPI 연동을 위한 generate_sections() 함수 제공
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.llms import Ollama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate

COLLECTION_NAME = "project_plan_chunks"
QDRANT_URL = "http://localhost:6333"
OLLAMA_MODEL = "qwen3:8b"

# 섹션별 검색 힌트 쿼리
SECTION_QUERY_HINTS = {
    "추진배경":         "{concept} 시장 현황 트렌드 문제 배경",
    "필요성":           "{concept} 기술적 경제적 사회적 필요성",
    "서비스 내용":      "{concept} 서비스 기능 특징 구현",
    "주요기술":         "{concept} 기술 스택 개발 방법론",
    "기대효과":         "{concept} 기대효과 성과 지표",
    "시장 현황":        "{concept} 시장 규모 동향 경쟁사",
    "문제점 및 개선방향": "{concept} 문제점 한계 개선 방향",
    "프로젝트 목표":    "{concept} 목표 달성 방향",
}

SECTION_PROMPT = PromptTemplate(
    input_variables=["section", "concept", "context"],
    template="""당신은 프로젝트 기획서 작성을 돕는 어시스턴트입니다.
아래 참고자료를 바탕으로 '{concept}' 프로젝트의 '{section}' 항목을 작성해주세요.

[작성 규칙]
- 참고자료의 내용을 기반으로 작성하되, 사실을 지어내지 말 것
- 구체적 수치가 필요한 경우 OO% 형식으로 표시할 것
- 한국어로 간결하고 명확하게 작성할 것
- /think 태그 없이 바로 본문만 출력할 것

[참고자료]
{context}

[작성할 항목: {section}]
"""
)


def build_rag_components():
    print("BGE-M3 임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
    )

    llm = Ollama(model=OLLAMA_MODEL)

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        llm=llm,
    )

    return retriever, llm


def generate_section(retriever, llm, section: str, concept: str) -> str:
    """섹션 하나에 대한 텍스트 생성"""
    hint = SECTION_QUERY_HINTS.get(section, f"{concept} {section}")
    query = hint.format(concept=concept)

    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = SECTION_PROMPT.format(
        section=section,
        concept=concept,
        context=context
    )

    return llm.invoke(prompt)


def generate_sections(concept: str, sections: list) -> dict:
    """
    FastAPI 연동용 메인 함수
    입력: concept(str), sections(list)
    출력: {섹션명: 생성된 텍스트} dict
    """
    retriever, llm = build_rag_components()
    results = {}

    for section in sections:
        print(f"생성 중: [{section}]")
        results[section] = generate_section(retriever, llm, section, concept)

    return results


def main():
    concept = "오픈소스 LLM과 RAG 기반 PDF 양식 자동 작성 시스템"
    test_sections = ["추진배경", "서비스 내용"]

    results = generate_sections(concept, test_sections)

    for section, text in results.items():
        print(f"\n{'='*60}")
        print(f"섹션: [{section}]")
        print("="*60)
        print(text)


if __name__ == "__main__":
    main()
