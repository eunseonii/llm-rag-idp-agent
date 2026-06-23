# 트러블슈팅 기록

RAG 파이프라인 개발 과정에서 겪은 기술적 문제와 해결 과정을 기록합니다.

---

## 6/15 (월) — 환경 세팅

### VS Code Remote-SSH 연결 실패

**문제:**
VS Code Remote-SSH로 Ubuntu VM(192.168.14.128)에 연결 시도했으나 "VS Code 서버를 초기화하는 중" 상태에서 멈추고 "SSH 연결이 끊어졌습니다" 에러 발생.

**원인 분석:**
1. cmd에서 ssh es@192.168.14.128 직접 연결 → 정상 동작 확인 → SSH 서버 자체는 문제없음
2. Remote-SSH 로그 확인 → 비밀번호 입력창이 뜨지 않거나 인식 안 되는 문제로 추정
3. VS Code 서버 파일 수동 설치 시도 → 커밋 ID 기반으로 서버 바이너리 직접 다운로드했으나 실패

**해결:**
비밀번호 방식 대신 SSH 키 인증으로 전환

    ssh-keygen -t rsa -b 4096
    type %USERPROFILE%\.ssh\id_rsa.pub | ssh es@192.168.14.128 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

**결과:**
SSH 키 인증 설정 후 VS Code Remote-SSH 연결 성공. 이후 비밀번호 입력 없이 자동 연결.

---

## 6/16 (화) — 초기 파이프라인 구축

### BGE-M3 vs TF-IDF 실험 (네트워크 제약 환경)

**문제:**
sentence-transformers로 BGE-M3 모델 다운로드 시도했으나 개발 환경의 네트워크가 Hugging Face에 접근 불가능.

**해결:**
실제 모델 없이도 전체 파이프라인 구조(청킹 → 벡터화 → Qdrant 저장 → 검색)를 검증하기 위해 scikit-learn의 TfidfVectorizer를 임시 대체로 사용. Qdrant 클라이언트는 동일하게 유지하고 벡터 차원만 300으로 조정 후, Ubuntu VM 환경에서 실제 BGE-M3로 교체.

**결과 비교:**

| 환경 | 임베딩 | 차원 | 검색 결과 |
|------|--------|------|-----------|
| 컨테이너 | TF-IDF | 300 | 헬스케어 문서 검색 실패 |
| Ubuntu VM | BGE-M3 | 1024 | 헬스케어 문서 정확히 검색 성공 |

**배운 점:**
단순 키워드 매칭(TF-IDF)과 의미 기반 임베딩(BGE-M3)의 실질적 차이를 수치로 확인. 한국어 특화 임베딩 모델의 필요성을 직접 증명하는 실험이 됨.

---

## 6/17 (수) — 코퍼스 전략 수립

### LLM 생성 데이터 의존도 문제

**문제:**
초기 코퍼스가 공공문서 3개 + LLM 생성 문서 4개로 구성되어 LLM 생성 데이터가 전체의 73%를 차지.

**원인 분석:**
    LLM 생성 문서 → RAG 검색 → 다시 LLM 입력
    = LLM의 환각이 RAG를 통해 증폭되는 구조
    = RAG의 핵심 가치인 "신뢰할 수 있는 외부 문서 기반 생성" 전제가 무너짐

**해결:**
LLM 생성 문서 전량 제외. 실제 프로젝트 기획서 PDF 3개 + SPRI AI 산업 동향 보고서 2개로 코퍼스 재구성. 총 1305개 청크 확보.

**배운 점:**
RAG는 검색 기반 시스템이므로 코퍼스 품질이 생성 품질을 직접 결정함. 데이터 양보다 신뢰도가 우선.

---

### 대상 양식 변경 (사업계획서 → 프로젝트 기획서)

**문제:**
초기 목표 양식인 사업계획서는 표 구조가 복잡해 PDF 자동 채움 정확도가 낮을 것으로 판단.

    중첩된 표 구조 → 셀 좌표 주입 오차 누적
    pdfplumber 인식 실패 케이스 다수
    한글 폰트 + 줄바꿈 + 셀 크기 오차 겹침

**해결:**
- 대상 양식을 프로젝트 기획서로 변경
- MVP 범위를 텍스트 섹션만 자동 채움으로 명확히 제한
- 표(간트차트, 역할분담 등)는 사용자 직접 수정 구간으로 분리

**배운 점:**
기술적 제약을 인정하고 MVP 범위를 현실적으로 재정의하는 것이 프로젝트 완성도를 높임.

---

## 6/18 (목) — LangChain 리팩토링

### 파이프라인 파일 구조 재설계

**문제:**
기획서에 LangChain 사용으로 명시했으나 실제 코드에 LangChain이 전혀 없었음.

    grep -rn "langchain" ~/vm_version/
    # 결과 없음

각 파일이 독립적으로 구현되어 파이프라인 연결이 비효율적이었고, FastAPI 통합 시 복잡도가 높아질 우려.

**해결:**
기존 4개 파일을 LangChain 기반 3개 파일로 리팩토링.

| 기존 | 변경 |
|------|------|
| 01_chunking.py | 01_load_and_chunk.py |
| 02_embedding_qdrant.py | 02_embed_and_store.py |
| 03_retrieve_generate.py | 03_rag_pipeline.py |
| 04_multi_query.py | 03_rag_pipeline.py에 통합 |

- DirectoryLoader + RecursiveCharacterTextSplitter로 문서 로딩 표준화
- HuggingFaceEmbeddings + QdrantVectorStore로 임베딩/저장 통합
- MultiQueryRetriever로 멀티쿼리 기능 내장화
- generate_sections() 함수로 FastAPI 연동 인터페이스 확립

**배운 점:**
프레임워크 도입은 코드를 줄이는 것뿐만 아니라 팀 간 인터페이스를 표준화하고 유지보수성을 높이는 역할을 함.

---

## 6/22 (월) — 코퍼스 교체 및 파이프라인 검증

### LangChain 1.x 버전 호환성 문제

**문제:**
LangChain 1.3.10 업그레이드 후 MultiQueryRetriever import 실패.

    from langchain.retrievers.multi_query import MultiQueryRetriever
    # ModuleNotFoundError: No module named 'langchain.retrievers'

**원인 분석:**
    find /home/es/.local/lib/python3.14/site-packages/langchain -name "*.py" | xargs grep -l "MultiQueryRetriever"
    # 결과 없음

    find /home/es/.local/lib/python3.14/site-packages -name "multi_query*"
    # /home/es/.local/lib/python3.14/site-packages/langchain_classic/retrievers/multi_query.py

LangChain 1.x에서 일부 모듈이 langchain_classic 패키지로 분리됨.

**해결:**
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever

---

### Ollama 모델 경로 문제 (VMware 공유 폴더 마운트 끊김)

**문제:**
VM 재시작 후 Ollama가 모델을 찾지 못함. 모델이 VMware 공유 폴더(/mnt/hgfs/tooktak/ollama-models)에 저장되어 있었는데 마운트가 끊겨 있었음.

**원인:**
VMware 공유 폴더는 VM 재시작 시 자동 마운트되지 않음. Ollama가 systemd 서비스로 자동 실행되는데 환경변수(OLLAMA_MODELS)가 설정되지 않아 기본 경로에서 모델을 찾지 못함.

**해결:**
    sudo mkdir -p /mnt/hgfs/tooktak
    sudo vmhgfs-fuse .host:/tooktak /mnt/hgfs/tooktak -o allow_other
    sudo systemctl edit ollama
    # [Service]
    # Environment="OLLAMA_MODELS=/mnt/hgfs/tooktak/ollama-models"
    sudo systemctl restart ollama

**결과:**
VM 재시작 후에도 Ollama가 공유 폴더의 모델을 자동으로 인식.

---

### PyMuPDFLoader 적용 (txt → PDF 직접 로딩)

**문제:**
기존 파이프라인이 txt 파일만 처리하도록 구현되어 있어 PDF 코퍼스를 직접 사용할 수 없었음.

**해결:**
    # 기존
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)

    # 변경
    from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
    loader = DirectoryLoader(DOCS_DIR, glob="**/*.pdf", loader_cls=PyMuPDFLoader)

**결과:**
- 로드된 문서: 8개(txt) → 468개(PDF 페이지 단위)
- 생성된 청크: 105개 → 1305개
- RAG + Qwen3:8b 텍스트 생성 검증 완료
