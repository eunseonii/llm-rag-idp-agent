# LLM·RAG 기반 지능형 문서처리(IDP) 에이전트 개발

> 팀 프로젝트 **뚝딱(TookTak)** 의 RAG 파이프라인 담당 파트입니다.
> 한국선급(KR) 2026 규정 문서를 기반으로 하이브리드 검색 + 리랭킹 파이프라인을 단계별로 구현하고 성능을 검증했습니다.

---

## 담당 역할

4인 팀 프로젝트에서 **RAG 파이프라인 전담 개발**

- KR 2026 규정 PDF 파싱 및 전처리 (PUA 문자 제거, 파서 이중화)
- BGE-M3 Dense+Sparse 임베딩 시도 및 CPU 환경 한계 분석 (v1)
- Granite-embedding + fastembed BM25 + jina-reranker ONNX 전환 (v2)
- 하이브리드 검색 구현 (Dense + Sparse + RRF 리랭킹)
- recall@k 기반 검색 품질 평가 및 두 버전 성능 비교
- Ollama + Qwen3:8b 로컬 LLM 연동

---

## 개발 배경

뚝딱은 빈 양식 PDF에 사내 도메인 문서를 근거로 내용을 자동 생성해주는 온프레미스 문서처리 도구입니다.

핵심은 **사전에 구축된 도메인 코퍼스에서 관련 내용을 정확하게 찾아오는 검색 품질**이었고, 단순 키워드 검색이 아닌 의미 기반 하이브리드 검색이 필요했습니다.

KR 규정처럼 수식·기호가 많고 전문 용어가 밀집된 문서에서 검색 품질을 높이기 위해 이 파이프라인을 단계별로 직접 구현하고 검증했습니다.

---

## 시스템 아키텍처

```
[사전 구축 단계 - 개발 시]
KR 규정 PDF 10개 (총 6,732청크)
        ↓
파싱 (PyMuPDF → pdfplumber 이중화)
→ PUA 문자 제거 → 500자 청킹 (overlap=50)
        ↓
임베딩 (BGE-M3 / Granite+BM25)
        ↓
Qdrant kr_rules_study 컬렉션 저장 (고정)

[서비스 실행 단계 - 사용자 요청 시]
사용자: 빈 양식 PDF + 컨셉/키워드 입력
        ↓
양식 필드 파악 → 동적 쿼리 생성
        ↓
사전 구축된 kr_rules_study 검색
→ Dense + Sparse + RRF → Reranker
        ↓
Qwen3:8b (Ollama) → 필드 내용 생성 → PDF 반환
```

> **설계 원칙:** 기반 지식은 사전에 구축된 코퍼스에서 가져오되, 사용자의 실시간 입력을 반영해 쿼리를 동적으로 생성하는 하이브리드 RAG 구조입니다.

---

## 버전 변천과 기술 선택 이유

### v1 — BGE-M3

Dense+Sparse를 단일 모델로 동시 추출할 수 있어 채택했으나, CPU 환경에서 배치당 60~70초, 전체 인덱싱에 약 3시간 47분 소요. WSL로 환경을 바꿔도 약 2시간으로 근본 해결이 안 되며 모델 자체(1024차원)가 무거운 것이 원인. 팀 운영 서버(Ollama 0.24)에서 NaN 버그도 확인.

### v2 — Granite + fastembed (진행 중)

| 항목 | v1 | v2 |
|------|----|----|
| 임베딩 | BGE-M3 1024d | Granite-embedding 768d (Ollama) |
| Sparse | BGE-M3 내장 | fastembed BM25 |
| 리랭커 | bge-reranker-v2-m3 | jina-reranker-v2 ONNX |

> 동일 조건(PDF 10개, 500자 청킹)에서 recall@k를 비교해 순수한 스택 차이로 인한 성능 변화를 측정할 예정입니다.

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 임베딩 | BGE-M3 (v1) / Granite-embedding + fastembed (v2) |
| 벡터DB | Qdrant (Docker) |
| 리랭커 | bge-reranker-v2-m3 (v1) / jina-reranker-v2 ONNX (v2) |
| LLM | Ollama + Qwen3:8b |
| PDF 처리 | PyMuPDF + pdfplumber |

---

## 파이프라인 구조

```
01_parse_and_chunk.py   PDF 파싱 + PUA 제거 + 청킹
        ↓
02_embed_and_store.py   임베딩 → Qdrant 저장
        ↓
03_hybrid_search.py     Dense + Sparse + RRF 하이브리드 검색
        ↓
04_evaluate.py          recall@k 평가 (v1 vs v2 비교)
        ↓
05_reranker_bench.py    리랭커 속도·분별력 벤치마크
```

---

## 코퍼스 구성

한국선급(KR) 2026 규정 PDF 10개를 직접 수집해 코퍼스로 사용했습니다.
출처: [KR Rules 2026](https://www.krs.co.kr/KRRules/KRRules2026/KRRulesK.html)

| 파일명 | 카테고리 | 청크 수 |
|--------|----------|---------|
| 1편_선급등록및검사.pdf | 선급및강선규칙 | 1,232 |
| 2편_재료및용접.pdf | 선급및강선규칙 | 1,014 |
| 3편_선체구조.pdf | 선급및강선규칙 | 813 |
| 6편_전기설비및제어시스템.pdf | 선급및강선규칙 | 491 |
| 7편_전용선박(5,6장).pdf | 선급및강선규칙 | 1,299 |
| 8편_방화및소화.pdf | 선급및강선규칙 | 634 |
| 이동식해양구조물규칙.pdf | 해양구조물규칙 | 313 |
| 자율운항선박지침.pdf | 기타기술규칙 | 75 |
| 저인화점연료선박규칙.pdf | 기타기술규칙 | 773 |
| 해상사이버보안시스템지침.pdf | 기타기술규칙 | 88 |
| **합계** | | **6,732** |

---

## 트러블슈팅 요약

뚝딱 프로젝트 RAG 파이프라인 담당으로 참여했습니다.
팀 서버 환경에서는 자동화로 넘어갔던 이슈들을,
개인 VM에서 동일한 파이프라인을 단계별로 직접 재현하며
하나씩 원인을 파악하고 해결했습니다.

| # | 문제 | 원인 | 해결 |
|---|------|------|------|
| 1 | Qdrant 400 에러 | PUA 문자(U+E000~U+F8FF) payload 포함 | 정규식으로 파싱 단계에서 제거 |
| 2 | curl 업로드 문법 오류 | 파일명 `()` 를 쉘이 서브쉘로 해석 | 작은따옴표로 경로 감싸기 |
| 3 | Qdrant 400 에러 | chunk_id가 문자열 (정수/UUID만 허용) | 전역 카운터로 정수 ID 부여 |
| 4 | 디스크 100% 초과 | GPU용 torch + 캐시 누적 | CPU 전용 torch 재설치, 캐시 정리 |

자세한 내용 → [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## 개발 현황

- [x] KR 2026 규정 PDF 코퍼스 구성
- [x] PDF 파싱 + 청킹 (PUA 문자 처리 포함) → 6,732청크 생성
- [x] v1 BGE-M3 임베딩 파이프라인 구현 및 속도 한계 확인
- [ ] v2 Granite + fastembed 임베딩 → Qdrant 인덱싱
- [ ] 하이브리드 검색 (Dense+Sparse+RRF) 구현
- [ ] 리랭커 도입 및 속도 최적화
- [ ] recall@k 기반 v1 vs v2 성능 비교
- [ ] RAG + LLM 텍스트 생성 동작 확인

---

## 관련 링크

- KR Rules 2026: https://www.krs.co.kr/KRRules/KRRules2026/KRRulesK.html
