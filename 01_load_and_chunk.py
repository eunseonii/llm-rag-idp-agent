# -*- coding: utf-8 -*-
"""
[1단계] 문서 로딩 & 청킹 (LangChain 기반)
- docs/ 폴더의 txt 파일들을 LangChain으로 로드
- RecursiveCharacterTextSplitter로 청킹
- chunks.json으로 저장
"""
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os

DOCS_DIR = "./corpus"
OUT_PATH = "./chunks.json"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def main():
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.pdf",
    	loader_cls=PyMuPDFLoader
    )
    documents = loader.load()
    print(f"로드된 문서: {len(documents)}개")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"생성된 청크: {len(chunks)}개")

    chunk_list = []
    for i, chunk in enumerate(chunks):
        chunk_list.append({
            "id": i,
            "source_file": os.path.basename(chunk.metadata.get("source", "")),
            "text": chunk.page_content,
            "metadata": chunk.metadata
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_list, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {OUT_PATH}")
    print("\n--- 청크 미리보기 (앞 3개) ---")
    for c in chunk_list[:3]:
        preview = c["text"][:80].replace("\n", " ")
        print(f"[{c['id']}] ({c['source_file']}) {preview}...")

if __name__ == "__main__":
    main()
