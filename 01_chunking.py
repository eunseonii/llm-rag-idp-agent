# -*- coding: utf-8 -*-
"""
[1단계] 청킹(Chunking)
- docs/ 폴더의 사업계획서 txt 파일들을 읽어서
  [1. 문제인식], [2. 실현가능성] 등 항목 단위로 쪼갠다.
- 너무 긴 항목은 문단(빈 줄) 기준으로 한 번 더 쪼갠다.

[VM 적용 안내]
이 파일은 외부 모델이나 인터넷 연결이 필요 없는 순수 텍스트 처리이므로
수정 없이 그대로 사용 가능하다. docs/ 폴더에 사업계획서 txt 파일들을
넣고 실행하면 chunks.json이 생성된다.
"""
import re
import os
import json

# ===== 환경에 맞게 경로만 확인/수정 =====
DOCS_DIR = "./docs"
OUT_PATH = "./chunks.json"
# =======================================

SECTION_PATTERN = re.compile(r"^\[(.+?)\]$", re.MULTILINE)
MAX_CHUNK_LEN = 400  # 한 청크 최대 글자 수


def split_into_sections(text: str):
    """파일 텍스트를 [헤더] 기준으로 섹션 단위로 분리"""
    matches = list(SECTION_PATTERN.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        header = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append({"header": header, "body": body})
    return sections


def split_long_section(body: str, max_len: int):
    """긴 섹션을 문단(빈 줄) 기준으로 추가 분할"""
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_len:
            current = (current + "\n" + p).strip()
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks if chunks else [body]


def main():
    all_chunks = []
    chunk_id = 0

    if not os.path.isdir(DOCS_DIR):
        raise FileNotFoundError(
            f"{DOCS_DIR} 폴더가 없습니다. 사업계획서 txt 파일들을 이 폴더에 넣어주세요."
        )

    files = sorted(f for f in os.listdir(DOCS_DIR) if f.endswith(".txt"))
    if not files:
        raise FileNotFoundError(f"{DOCS_DIR} 안에 .txt 파일이 없습니다.")

    for fname in files:
        path = os.path.join(DOCS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        sections = split_into_sections(text)
        title_match = re.search(r"\[창업아이템명\]\s*(.+)", text)
        item_title = title_match.group(1).strip() if title_match else fname

        for sec in sections:
            sub_chunks = split_long_section(sec["body"], MAX_CHUNK_LEN)
            for sc in sub_chunks:
                all_chunks.append({
                    "id": chunk_id,
                    "source_file": fname,
                    "item_title": item_title,
                    "section": sec["header"],
                    "text": sc,
                })
                chunk_id += 1

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"총 {len(files)}개 파일 처리")
    print(f"총 {len(all_chunks)}개 청크 생성")
    print(f"저장 위치: {OUT_PATH}")

    print("\n--- 청크 미리보기 (앞 5개) ---")
    for c in all_chunks[:5]:
        preview = c["text"][:60].replace("\n", " ")
        print(f"[{c['id']}] ({c['source_file']} / {c['section']}) {preview}...")


if __name__ == "__main__":
    main()
