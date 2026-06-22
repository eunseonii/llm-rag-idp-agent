# -*- coding: utf-8 -*-
"""
[2단계] 임베딩 & Qdrant 저장 (LangChain + BGE-M3)
- LangChain HuggingFaceEmbeddings로 BGE-M3 임베딩
- QdrantVectorStore로 저장
- 저장 후 검색 테스트
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

DOCS_DIR = "./docs"
COLLECTION_NAME = "project_plan_chunks"
QDRANT_URL = "http://localhost:6333"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def main():
    # 문서 로드
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"로드된 문서: {len(documents)}개")

    # 청킹
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"생성된 청크: {len(chunks)}개")

    # BGE-M3 임베딩
    print("BGE-M3 임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 기존 컬렉션 삭제 후 재생성
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"기존 컬렉션 '{COLLECTION_NAME}' 삭제")

    # Qdrant 저장
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )
    print(f"Qdrant 저장 완료: {len(chunks)}개 청크")

    # 테스트 검색
    print("\n--- 검색 테스트 ---")
    results = vectorstore.similarity_search("프로젝트 추진 배경 및 필요성", k=3)
    for i, r in enumerate(results):
        print(f"[{i+1}] {r.page_content[:80].replace(chr(10), ' ')}...")

if __name__ == "__main__":
    main()
