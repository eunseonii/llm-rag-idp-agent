# LLM·RAG 기반 지능형 문서처리(IDP) 에이전트 개발

> 팀 프로젝트 **뚝딱(TookTak)** 의 RAG 파이프라인 담당 파트입니다.
> 한국선급(KR) 2026 규정 문서를 기반으로 하이브리드 검색 + 리랭킹 파이프라인을 구현했습니다.

---

## 담당 역할

4인 팀 프로젝트에서 **RAG 파이프라인 전담 개발**

- KR 2026 규정 PDF 파싱 및 전처리 (PUA 문자 제거, 파서 이중화)
- BGE-M3 Dense+Sparse 임베딩 + Qdrant 벡터 저장소 구축
- 하이브리드 검색 구현 (Dense + Sparse + RRF 리랭킹)
- BGE-Reranker 기반 크로스인코더 리랭킹 도입
- LangChain 기반 RAG 파이프라인 설계 및 구현
- Ollama + Qwen3:8b 로컬 LLM 연동
- FastAPI 연동용 `generate_sections()` 인터페이스 설계

---

## 왜 만들었나

뚝딱은 빈 양식 PDF에 참고자료를 기반으로 내용을 자동 생성해주는 온프레미스 문서처리 도구입니다. 핵심은 참고자료에서 관련 내용을 정확하게 찾아오는 검색 품질이었고, 단순 키워드 검색이 아닌 의미 기반 하이브리드 검색이 필요했습니다.

KR 규정처럼 수식·기호가 많고 전문 용어가 밀집된 문서에서 검색 품질을 높이기 위해 이 파이프라인을 직접 설계하고 구현했습니다.

---

## 시스템 아키텍처

```
[사용자 업로드]
빈 양식 PDF + 참고자료 PDF + 컨셉 입력
        ↓
[전처리]                                        
참고자료 파싱 (PyMuPDF → pdfplumber 이중화)
→ PUA 문자 제거 → 500자 청킹 (overlap=50)
        ↓
[Qdrant - 동적 인덱싱]                          
BGE-M3 Dense+Sparse 임베딩
→ user_{id} 컬렉션 생성 → 저장 (세션 유지)
        ↓
[하이브리드 검색 + Reranker]                    
양식 필드별 쿼리
→ Dense + Sparse + RRF → BGE-Reranker
        ↓
[Qwen3:8b 생성]
컨텍스트 + 컨셉 → 섹션 텍스트 생성 (Ollama)
        ↓
[PostgreSQL - 이력 저장]
생성 결과 + 메타데이터 + user_id
+ Qdrant 컬렉션 참조 저장
        ↓
[PDF 출력]
완성된 PDF 반환
```

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 임베딩 | BGE-M3 (Dense + Sparse) |
| 벡터DB | Qdrant (Docker) |
| 검색 | 하이브리드 검색 (RRF) + BGE-Reranker |
| LLM | Ollama + Qwen3:8b |
| PDF 처리 | PyMuPDF + pdfplumber |
| 백엔드 | FastAPI |

---

## 파이프라인 구조

```
01_parse_and_chunk.py   PDF 파싱 + PUA 제거 + 청킹
        ↓
02_embed_and_store.py   BGE-M3 Dense+Sparse 임베딩 → Qdrant 저장
        ↓
03_hybrid_search.py     Dense + Sparse + RRF 하이브리드 검색
        ↓
04_evaluate.py          recall@k 평가
        ↓
05_reranker.py          BGE-Reranker 크로스인코더 리랭킹
        ↓
06_rag_pipeline.py      RAG + LLM 텍스트 생성 (FastAPI 연동)
```

---

## 파일 구조

```
kr-rules-rag/
├── src/
│   ├── 01_parse_and_chunk.py
│   ├── 02_embed_and_store.py
│   ├── 03_hybrid_search.py
│   ├── 04_evaluate.py
│   ├── 05_reranker.py
│   ├── 06_rag_pipeline.py
│   └── app/main.py
├── data/
│   ├── raw/                    KR 규정 PDF 원본 10개
│   └── chunks/                 파싱 결과 JSONL (총 6,732청크)
├── docs/
│   ├── TROUBLESHOOTING.md
│   └── development-log.md
└── scripts/
    └── download_corpus.sh
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

> LLM 생성 문서는 코퍼스에서 제외했습니다. 환각 내용이 검색 결과에 섞이면 생성 품질이 떨어지기 때문입니다.

---

## 트러블슈팅

개발 과정에서 직접 발견하고 해결한 이슈들입니다.

### 1. PUA 문자로 인한 임베딩 400 에러

**증상:** 임베딩 API 호출 시 400 에러 발생

**원인:** PDF 제작 도구가 수식 기호를 유니코드 사적 영역(U+E000~U+F8FF)에 임의 매핑. 3편_선체구조.pdf에서 `규칙길이(\ue00b)`, `흘수(\ue0e8\ue0f7)` 처럼 추출됨. 10개 중 8개 파일에서 발견, 3편이 23,937개로 최다.

**해결:** 파싱 단계에서 PUA 문자 제거

```python
PUA_PATTERN = re.compile(r"[\uE000-\uF8FF]")
cleaned = PUA_PATTERN.sub("", text)
```

### 2. 파일명 괄호로 인한 curl 업로드 실패

**증상:**
```
bash: 예기치 않은 `(' 토큰 주변에서 문법 오류
```

**원인:** `7편_전용선박(5,6장).pdf` 파일명의 `()`를 쉘이 서브쉘 토큰으로 해석

**해결:** 작은따옴표로 경로 전체를 감싸기

```bash
# 실패
curl -F file=@7편_전용선박(5,6장).pdf http://localhost:8000/upload

# 성공
curl -F 'file=@7편_전용선박(5,6장).pdf' http://localhost:8000/upload
```

### 3. 리랭커 속도 문제 (CPU 환경)

BGE-Reranker-v2-m3(568MB)가 CPU에서 ~37초/질의로 실사용 불가. BGE-Reranker-base(278MB)로 교체 후 ~9초로 단축. `fetch_k=12`, `max_len=256`, `threads=16` 고정.

자세한 내용 → [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## 개발 현황

- [x] KR 2026 규정 PDF 코퍼스 구성
- [x] PDF 파싱 + 청킹 (PUA 문자 처리 포함) → 6,732청크 생성
- [ ] BGE-M3 Dense+Sparse 임베딩 + Qdrant 인덱싱
- [ ] 하이브리드 검색 (Dense+Sparse+RRF) 구현
- [ ] BGE-Reranker 도입 및 속도 최적화
- [ ] RAG + LLM 텍스트 생성 동작 확인
- [ ] FastAPI 연동
- [ ] 실제 PDF 양식 필드 매핑 검증
- [ ] 프롬프트 최적화 (수치 환각 억제)

---

## 관련 링크

- KR Rules 2026: https://www.krs.co.kr/KRRules/KRRules2026/KRRulesK.html
