"""
01_parse_and_chunk.py
=====================
KR 규정 PDF 10개를 파싱하고 500자 청크로 분할해서 JSONL로 저장합니다.

[왜 fitz(PyMuPDF)를 먼저 쓰나요?]
  - fitz는 C 기반이라 빠르고 대부분의 PDF에서 텍스트를 잘 뽑습니다.
  - 단, 수식이 많거나 레이아웃이 복잡한 PDF에선 텍스트가 깨지거나
    거의 안 나오는 경우가 있어요.

[왜 pdfplumber를 fallback으로 쓰나요?]
  - pdfplumber는 좌표 기반으로 텍스트를 추출해서 레이아웃 복잡한 PDF에
    강점이 있습니다.
  - 단, fitz보다 느려서 1차 시도에는 부적합해요.

[PUA(Private Use Area)가 뭔가요?]
  - 유니코드 U+E000~U+F8FF 구간은 "사용 목적을 정하지 않은" 사적 영역입니다.
  - PDF 편집 소프트웨어들이 특수기호(수식 기호, 도형 등)를 이 구간에
    임의로 배치하는 경우가 많아요.
  - 임베딩 모델(BGE-M3 등)에 이 문자를 그대로 보내면 400 에러가 납니다.
    → 제거해야 합니다.

[청크 사이즈 500, overlap 50으로 정한 이유]
  - RAG에서 청크가 너무 크면 관련 없는 내용이 섞이고,
    너무 작으면 문맥이 잘려서 답변 품질이 떨어집니다.
  - 500자는 한국어 기준 약 3~5문장, KR 규정 조항 하나 정도의 크기예요.
  - overlap=50은 청크 경계에서 문장이 잘리는 것을 보완합니다.
"""

import fitz          # PyMuPDF — pip install pymupdf
import pdfplumber    # fallback 추출기
import re
import json
import logging
from pathlib import Path

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
# __file__은 이 스크립트 자체의 경로입니다.
# .parent로 src/ 폴더를 가리키고, 한 단계 더 올라가서 프로젝트 루트를 잡아요.
BASE_DIR   = Path(__file__).parent.parent          # ~/kr-rules-rag/
DATA_DIR   = BASE_DIR / "data" / "raw"             # PDF 파일들이 있는 곳
OUT_DIR    = BASE_DIR / "data" / "chunks"          # 결과 JSONL 저장 위치
LOG_DIR    = BASE_DIR / "logs"

# 출력 폴더가 없으면 만들어 둡니다.
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 로깅 설정
# ──────────────────────────────────────────────
# 로그를 파일 + 콘솔 양쪽에 출력합니다.
# 디버깅할 때 콘솔에서 실시간으로 보고, 나중엔 파일로 돌아볼 수 있어요.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "01_parse.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 설정값 (한 곳에서 관리)
# ──────────────────────────────────────────────
CHUNK_SIZE    = 500   # 청크 최대 글자 수 (한국어 기준 약 3~5문장)
OVERLAP       = 50    # 청크 간 겹치는 글자 수
MIN_TEXT_LEN  = 100   # 이것보다 짧으면 fitz 추출 실패로 간주 → pdfplumber 재시도
PUA_PATTERN   = re.compile(r"[\uE000-\uF8FF]")  # Private Use Area 유니코드 범위

# ──────────────────────────────────────────────
# 함수 1: PUA 문자 제거
# ──────────────────────────────────────────────
def remove_pua(text: str) -> str:
    """
    PUA(Private Use Area) 문자를 제거합니다.

    [면접 포인트]
    Q: "왜 PUA 문자가 문제가 되나요?"
    A: PDF 제작 도구들이 수식 기호를 유니코드 사적 영역(U+E000-U+F8FF)에
       임의 매핑하는 경우가 있습니다. 이 문자들은 의미 없는 노이즈이고,
       일부 임베딩 API에서 처리 불가로 400 에러를 냅니다.
       re.sub으로 해당 범위 전체를 빈 문자열로 교체해서 해결합니다.
    """
    cleaned = PUA_PATTERN.sub("", text)
    return cleaned


# ──────────────────────────────────────────────
# 함수 2: fitz로 PDF 텍스트 추출 (1차)
# ──────────────────────────────────────────────
def extract_with_fitz(pdf_path: Path) -> str:
    """
    PyMuPDF(fitz)로 PDF 전체 텍스트를 추출합니다.

    [왜 page.get_text("text")를 쓰나요?]
    - "text" 모드는 읽기 순서대로 텍스트를 이어붙여 줍니다.
    - "dict" 모드는 좌표 정보까지 주지만 후처리가 복잡해져요.
    - 우리 목적(RAG용 텍스트 덩어리)엔 "text"로 충분합니다.
    """
    text_parts = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            text_parts.append(page_text)
        doc.close()
    except Exception as e:
        log.warning(f"fitz 실패 ({pdf_path.name}): {e}")
        return ""

    return "\n".join(text_parts)


# ──────────────────────────────────────────────
# 함수 3: pdfplumber로 PDF 텍스트 추출 (fallback)
# ──────────────────────────────────────────────
def extract_with_pdfplumber(pdf_path: Path) -> str:
    """
    pdfplumber로 PDF 전체 텍스트를 추출합니다.
    fitz에서 텍스트가 거의 안 나왔을 때 재시도용으로 사용합니다.

    [왜 pdfplumber가 fallback인가요?]
    - pdfplumber는 좌표 기반 추출로 표나 다단 레이아웃에 강하지만,
      단순 텍스트 PDF에선 fitz보다 약 3~5배 느립니다.
    - 항상 쓰기엔 느리고, 꼭 필요한 경우에만 쓰는 게 효율적이에요.
    """
    text_parts = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
    except Exception as e:
        log.warning(f"pdfplumber 실패 ({pdf_path.name}): {e}")
        return ""

    return "\n".join(text_parts)


# ──────────────────────────────────────────────
# 함수 4: 텍스트 → 청크 리스트로 분할
# ──────────────────────────────────────────────
def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """
    긴 텍스트를 일정 크기의 청크로 분할합니다.

    [슬라이딩 윈도우 방식이란?]
    - start=0에서 시작해서 chunk_size만큼 잘라내고,
      다음 시작점은 (chunk_size - overlap)만큼 앞으로 당깁니다.
    - 예: size=500, overlap=50이면 0~500, 450~950, 900~1400 ...
    - 경계에서 문장이 잘리더라도 overlap 구간에서 복구됩니다.

    [왜 단순 split()이 아닌 슬라이딩 윈도우를 쓰나요?]
    - split()은 경계에서 문맥이 완전히 끊깁니다.
    - 슬라이딩 윈도우는 앞 청크의 끝부분을 다음 청크가 조금 가져가서
      검색 시 경계 근처 정보도 찾을 수 있습니다.
    """
    chunks = []
    text_len = len(text)
    start = 0
    step = chunk_size - overlap  # 실제로 앞으로 이동하는 크기

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if len(chunk) > 20:  # 너무 짧은 조각(공백, 페이지 번호 등)은 제외
            chunks.append(chunk)
        start += step

    return chunks


# ──────────────────────────────────────────────
# 함수 5: 버그1 재현 — PUA 문자 샘플 출력
# ──────────────────────────────────────────────
def demo_bug1_pua(raw_text: str, filename: str):
    """
    버그1 재현: PUA 문자가 텍스트에 얼마나 있는지 보여줍니다.

    [면접 포인트]
    Q: "어떻게 PUA 버그를 발견했나요?"
    A: 2편_재료및용접.pdf처럼 수식이 많은 PDF를 추출했더니
       텍스트에 의미 없는 □ 또는 ? 같은 문자가 다수 섞여 있었고,
       실제 유니코드 코드포인트를 확인하니 E000~F8FF 범위였습니다.
       이 상태로 임베딩 API에 보내면 400 에러가 발생했습니다.
    """
    pua_chars = PUA_PATTERN.findall(raw_text)
    if pua_chars:
        sample = raw_text[:300]  # 앞부분 300자만 보여줌
        log.warning(
            f"[버그1 재현] {filename}: PUA 문자 {len(pua_chars)}개 발견!\n"
            f"  코드포인트 샘플: {[hex(ord(c)) for c in set(pua_chars[:10])]}\n"
            f"  텍스트 앞부분 미리보기:\n  {repr(sample[:200])}"
        )
    else:
        log.info(f"[버그1] {filename}: PUA 문자 없음")


# ──────────────────────────────────────────────
# 함수 6: PDF 한 파일 전체 처리
# ──────────────────────────────────────────────
def process_pdf(pdf_path: Path) -> list[dict]:
    """
    PDF 한 파일을 파싱 → 정제 → 청킹까지 처리합니다.
    반환값은 청크 딕셔너리의 리스트입니다.

    반환 딕셔너리 구조:
    {
        "chunk_id":   "파일명_0001" 형태의 고유 ID,
        "source":     원본 파일명,
        "chunk_index": 몇 번째 청크인지,
        "text":       실제 텍스트,
        "char_count": 글자 수
    }
    """
    filename = pdf_path.name
    log.info(f"처리 시작: {filename}")

    # ── 1차: fitz 추출
    raw_text = extract_with_fitz(pdf_path)
    extractor_used = "fitz"

    # ── fitz 결과가 빈약하면 pdfplumber로 재시도
    # [왜 MIN_TEXT_LEN=100으로 잡았나요?]
    # 100자 미만이면 목차/표지만 있거나 이미지 PDF일 가능성이 높습니다.
    if len(raw_text.strip()) < MIN_TEXT_LEN:
        log.warning(f"fitz 결과 빈약 ({len(raw_text)}자) → pdfplumber 재시도: {filename}")
        raw_text = extract_with_pdfplumber(pdf_path)
        extractor_used = "pdfplumber"

    log.info(f"  추출 완료 ({extractor_used}): {len(raw_text):,}자")

    # ── 버그1 재현: PUA 문자 확인
    demo_bug1_pua(raw_text, filename)

    # ── PUA 제거 (버그1 수정)
    cleaned_text = remove_pua(raw_text)
    removed_count = len(raw_text) - len(cleaned_text)
    if removed_count > 0:
        log.info(f"  PUA 제거: {removed_count}자 제거됨")

    # ── 기본 노이즈 제거
    # 연속 공백/개행을 단일화합니다.
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)  # 빈 줄 3개 이상 → 2개
    cleaned_text = re.sub(r"[ \t]{2,}", " ", cleaned_text)  # 연속 공백 → 1개

    # ── 청킹
    chunks_text = split_into_chunks(cleaned_text)
    log.info(f"  청킹 완료: {len(chunks_text)}개 청크 (목표 크기 {CHUNK_SIZE}자)")

    # ── 딕셔너리로 구조화
    # chunk_id는 나중에 Qdrant 포인트 ID로도 활용합니다.
    stem = pdf_path.stem  # 확장자 제거한 파일명
    records = []
    for idx, chunk_text in enumerate(chunks_text):
        records.append({
            "chunk_id":    f"{stem}_{idx:04d}",  # 예: 2편_재료및용접_0001
            "source":      filename,
            "chunk_index": idx,
            "text":        chunk_text,
            "char_count":  len(chunk_text),
        })

    return records


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────
def main():
    # PDF 파일 목록 수집
    # [왜 sorted()를 쓰나요?]
    # 파일 시스템에서 glob() 결과 순서는 OS마다 다릅니다.
    # 정렬해두면 재현 가능하고, 로그 보기도 편해요.
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        log.error(f"PDF 파일이 없습니다: {DATA_DIR}")
        return

    log.info(f"총 {len(pdf_files)}개 PDF 발견")

    # ── 버그2 재현: 쉼표 파일명 경고
    # [왜 curl에서 쉼표가 문제가 되나요?]
    # curl의 --form 또는 -F 옵션에서 파일 경로에 쉼표가 있으면
    # curl이 이를 "여러 값 구분자"로 해석해서 파싱 오류가 납니다.
    # 예: curl -F "file=@7편(5,6장).pdf"  ← 이렇게 따옴표 없이 쓰면 실패
    # 수정: curl -F 'file=@7편(5,6장).pdf'  ← 작은따옴표로 감싸기
    for pdf_path in pdf_files:
        if "," in pdf_path.name:
            log.warning(
                f"[버그2 재현] 쉼표 포함 파일명 발견: '{pdf_path.name}'\n"
                f"  → curl 업로드 시 반드시 작은따옴표로 감싸야 합니다.\n"
                f"  잘못된 예: curl -F \"file=@{pdf_path.name}\"\n"
                f"  올바른 예: curl -F 'file=@{pdf_path.name}'"
            )

    # ── 파일별 처리
    total_chunks = 0
    for pdf_path in pdf_files:
        records = process_pdf(pdf_path)

        # JSONL 저장 (한 줄 = 청크 하나)
        # [왜 JSONL인가요?]
        # JSON 배열은 파일 전체를 메모리에 올려야 읽을 수 있습니다.
        # JSONL(JSON Lines)은 한 줄씩 읽을 수 있어서 대용량에 유리해요.
        out_path = OUT_DIR / f"{pdf_path.stem}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        total_chunks += len(records)
        log.info(f"  저장 완료: {out_path.name} ({len(records)}청크)\n")

    log.info(f"=== 완료: 전체 {total_chunks}개 청크 생성 ===")
    log.info(f"출력 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
