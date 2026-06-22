# RAG 파이프라인 - VM 실행 가이드

이 폴더는 컨테이너 테스트에서 검증한 코드 구조를, 실제 VM(Ubuntu)에서
BGE-M3 + Qdrant(Docker)로 그대로 동작시키기 위한 버전입니다.

## 폴더 구성
```
vm_version/
├── docs/                          # 사업계획서 텍스트 자료 7건
│   ├── 01_헬스케어.txt
│   ├── 02_푸드테크.txt
│   ├── 03_에듀테크.txt
│   ├── 04_펫테크.txt
│   ├── 05_정부_융합서비스_Xray.txt
│   ├── 06_정부_제조_자성분말.txt
│   └── 07_정부_지식서비스_부모제어앱.txt
├── 01_chunking.py                 # 1단계: 청킹 (수정 없이 그대로 사용)
└── 02_embedding_qdrant_search.py  # 2~4단계: 임베딩+적재+검색 (BGE-M3, Docker Qdrant)
```

## 실행 순서

### 0. 사전 준비
```bash
# 필요 패키지 설치
pip install sentence-transformers qdrant-client

# Qdrant를 Docker로 띄우기 (별도 터미널에서 실행, 계속 켜둘 것)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 1. 청킹 실행
```bash
cd vm_version
python3 01_chunking.py
```
- docs/ 폴더의 txt 7개를 읽어서 chunks.json 생성
- 정상 실행되면 "총 45개 청크 생성" 비슷한 메시지가 나옴 (텍스트를 추가하면 숫자는 달라짐)

### 2. 임베딩 + Qdrant 적재 + 검색 테스트
```bash
python3 02_embedding_qdrant_search.py
```
- BGE-M3 모델을 처음 실행 시 자동 다운로드 (약 2GB, 시간 소요 가능)
- Qdrant Docker 서버에 벡터 저장
- 미리 넣어둔 5개 테스트 쿼리로 검색 결과 출력

## 컨테이너 테스트와 다른 점

| 구분 | 컨테이너 테스트 | VM 버전 (이 폴더) |
|------|----------------|-------------------|
| 임베딩 모델 | TF-IDF (sklearn, 대체용) | BGE-M3 (실제 모델) |
| 벡터 차원 | 300 | 1024 |
| Qdrant 실행 방식 | 인메모리(`:memory:`) | Docker 서버 (데이터 영구 보존) |

## 자주 발생할 수 있는 문제

**Q. "Connection refused" 에러가 난다**
→ Qdrant Docker가 안 떠 있는 경우입니다. `docker ps`로 컨테이너가 실행 중인지 확인하세요.

**Q. BGE-M3 다운로드가 너무 오래 걸린다 / 안 된다**
→ 인터넷 연결 상태를 확인하세요. 한 번 받으면 `~/.cache/huggingface` 폴더에 캐시되어 다음부터는 빠릅니다.

**Q. 검색 결과가 기대와 다르게 나온다**
→ 정상입니다. 지금 자료(7건, 45개 청크)는 최소 테스트용입니다. 자료를 늘리거나(LLM 생성 자료 추가),
   청킹 기준(MAX_CHUNK_LEN)을 조정하면서 결과를 관찰하고 튜닝하는 과정이 필요합니다.

## 다음 단계 (이후 진행할 것)
검색된 청크를 실제 LLM(Ollama + Qwen3)에 프롬프트로 넘겨서 PDF 양식 항목을
자동으로 채우는 단계로 이어집니다. 이 부분은 LLM 담당자와 함께 진행하면 됩니다.
