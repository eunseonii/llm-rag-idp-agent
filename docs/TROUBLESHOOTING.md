# 트러블슈팅 상세 기록

뚝딱 프로젝트 RAG 파이프라인 담당으로 참여했습니다.
팀 서버 환경에서는 자동화로 넘어갔던 이슈들을,
개인 VM에서 동일한 파이프라인을 단계별로 직접 재현하며
하나씩 원인을 파악하고 해결했습니다.

| # | 문제 | 단계 | 상태 |
|---|------|------|------|
| 1 | BGE-M3 vs TF-IDF 비교 실험 | 임베딩 검증 | 완료 |
| 2 | LLM 생성 데이터 코퍼스 오염 문제 | 코퍼스 설계 | 해결 |
| 3 | PUA 문자로 인한 Qdrant 400 에러 | 파싱 | 해결 |
| 4 | 파일명 괄호로 인한 쉘 문법 오류 | 환경 | 해결 |
| 5 | chunk_id 문자열로 인한 Qdrant 400 에러 | 임베딩 | 해결 |
| 6 | 디스크 100% 초과로 인한 패키지 설치 실패 | 환경 | 해결 |

---

## #1 BGE-M3 vs TF-IDF 비교 실험

### 문제 상황

초기 개발 환경의 네트워크가 HuggingFace에 접근 불가능한 상황에서, 전체 파이프라인 구조(청킹 → 벡터화 → Qdrant 저장 → 검색)를 먼저 검증해야 했습니다. 실제 임베딩 모델 없이도 파이프라인이 동작하는지 확인이 필요했습니다.

### 재현 과정

scikit-learn의 TfidfVectorizer를 BGE-M3 대체로 사용해 파이프라인을 구성했습니다.

```python
# TF-IDF로 임시 대체
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=300)
vectors = vectorizer.fit_transform(texts).toarray()
# Qdrant 저장은 동일하게 유지 (차원만 300으로 조정)
```

이후 Ubuntu VM 환경에서 실제 BGE-M3로 교체해 동일한 질의로 검색 결과를 비교했습니다.

### 원인 분석

TF-IDF와 의미 기반 임베딩의 근본적인 차이에서 발생했습니다.

```
TF-IDF
→ 단어 출현 빈도 기반
→ "방화구역"과 "화재 방지 구역"을 완전히 다른 문서로 처리
→ 동의어, 유사 표현 검색 불가

BGE-M3
→ 의미 기반 다국어 임베딩 (1024차원)
→ 문맥과 의미를 벡터 공간에 표현
→ 유사 의미 표현도 가까운 벡터로 매핑
```

### 해결 과정

파이프라인 구조 검증 후 BGE-M3로 교체했습니다.
이후 v1 완성 과정에서 BGE-M3의 CPU 속도 한계(배치당 70초, 총 3h47m)와
팀 운영 서버(Ollama 0.24)에서의 NaN 버그를 확인해
v2에서 Granite-embedding(768차원, Ollama)으로 최종 전환을 결정했습니다.

### 검증

| 환경 | 임베딩 | 차원 | KR 규정 검색 결과 |
|------|--------|------|-----------|
| TF-IDF 임시 적용 | TF-IDF | 300 | 동의어·유사표현 검색 불가 (키워드 일치만 검색) |
| BGE-M3 v1 | BGE-M3 | 1024 | 관련 문서 정확히 검색 성공 |
| Granite v2 (예정) | Granite | 768 | recall@k 비교 측정 예정 |

### 배운 점

단순 키워드 매칭(TF-IDF)과 의미 기반 임베딩(BGE-M3)의 실질적 차이를 수치로 직접 확인했습니다. KR 규정처럼 전문 용어가 밀집된 문서에서는 의미 기반 임베딩이 필수적입니다. 또한 파이프라인 구조 검증과 모델 성능 검증을 분리해 단계적으로 접근한 것이 개발 효율을 높였습니다.

---

## #2 LLM 생성 데이터 코퍼스 오염 문제

### 문제 상황

초기 코퍼스 구성 시 실제 문서가 부족해 LLM으로 생성한 문서를 코퍼스에 포함했습니다.
이 구성에서 LLM 생성 문서가 전체 코퍼스의 73%를 차지했습니다.

### 재현 과정

LLM 생성 문서가 포함된 코퍼스로 RAG 파이프라인을 실행하고
동일 질의에 대한 검색 결과와 생성 결과를 비교했습니다.

### 원인 분석

```
LLM 생성 문서 → RAG 검색 → 다시 LLM 입력
= LLM의 환각이 RAG를 통해 증폭되는 구조
= RAG의 핵심 가치인 "신뢰할 수 있는 외부 문서 기반 생성" 전제가 무너짐
```

RAG는 LLM이 모르는 내용을 외부 문서에서 찾아와 근거로 삼는 구조입니다.
LLM이 생성한 문서를 코퍼스로 쓰면 LLM 자신의 환각을 다시 근거로 참조하는 순환 오류가 발생합니다.

### 해결 과정

LLM 생성 문서 전량을 코퍼스에서 제외했습니다.
이후 신뢰할 수 있는 공식 문서(KR 규정 PDF 10개)만으로 코퍼스를 재구성했습니다.

```
제외: LLM 생성 문서 (전체의 73%)
유지: 실제 공식 문서만
재구성: KR 규정 PDF 10개 → 6,732청크
```

### 검증

LLM 생성 문서 제거 후 동일 질의("방화구역 설치 기준")에 대한 검색 결과가
실제 KR 규정 내용을 근거로 정확하게 반환되는 것을 확인했습니다.

### 배운 점

RAG 시스템에서 코퍼스 품질은 생성 품질을 직접 결정합니다.
데이터 양보다 신뢰도가 우선이며, 코퍼스 설계 단계에서
"이 문서를 근거로 삼아도 되는가"를 반드시 검증해야 합니다.
특히 LLM 생성 데이터는 환각 증폭 위험이 있어 코퍼스에서 제외하는 것이 원칙입니다.

---

## #3 PUA 문자로 인한 Qdrant 400 에러

### 문제 상황

KR 규정 PDF에서 텍스트를 추출하면 수식·기호 영역이 유니코드 사적 영역(PUA, U+E000~U+F8FF) 문자로 깨져 나옵니다. 이 문자가 payload에 포함된 채로 Qdrant에 전송되면 JSON 파싱 실패로 400 에러가 발생합니다.

### 재현 과정

`fix_pua()` 함수를 의도적으로 비활성화한 상태로 임베딩 파이프라인을 실행했습니다.

```python
# fix_pua() 비활성화 상태 (버그 재현용)
text = chunk["text"]
# text = fix_pua(text)  ← 주석 처리
```

### 원인 분석

PDF 제작 도구가 수식 기호를 PUA 영역에 임의 매핑합니다.
3편_선체구조.pdf에서 `규칙길이(\ue00b)`, `흘수(\ue0e8\ue0f7)` 형태로 추출됩니다.
KR 규정 PDF 10개 중 8개에서 발견되었으며, 3편이 23,937개로 최다였습니다.

가설 검증 과정:
- 가설 1: 임베딩 모델이 PUA 문자를 처리하지 못해서 → 틀림 (모델은 정상 처리)
- 가설 2: Qdrant JSON 전송 시 파싱 실패 → 정답 (에러 메시지에서 확인)

### 해결 과정

파싱 단계에서 정규식으로 PUA 문자를 제거했습니다.

```python
import re

PUA_PATTERN = re.compile(r"[\uE000-\uF8FF]")

def fix_pua(text: str) -> str:
    return PUA_PATTERN.sub("", text)
```

임베딩 단계가 아닌 파싱 단계에서 제거한 이유는, 오염된 텍스트가 청크에 저장되면 나중에 검색 결과로 나올 때 LLM에 깨진 문자가 그대로 전달되기 때문입니다.

### 검증

fix_pua() 적용 후 재실행하여 400 에러 없이 전체 6,732청크 저장 완료를 확인했습니다.

### 배운 점

PDF는 화면 렌더링용 포맷이라 텍스트 추출 시 폰트 매핑 오류가 빈번합니다. 특히 수식·기호가 많은 기술 문서일수록 PUA 문자 비율이 높아 전처리 단계에서 반드시 처리해야 합니다. 문제를 임베딩 단계에서 잡으려 하지 않고 파싱 단계에서 원천 차단하는 것이 올바른 설계입니다.

---

## #4 파일명 괄호로 인한 쉘 문법 오류

`7편_전용선박(5,6장).pdf` 파일명의 `()`를 bash가 서브쉘 토큰으로 해석해 문법 오류 발생.

```bash
# 실패
curl -F file=@7편_전용선박(5,6장).pdf http://localhost:8000/upload

# 성공 - 작은따옴표로 경로 전체를 감싸기
curl -F 'file=@7편_전용선박(5,6장).pdf' http://localhost:8000/upload
```

실제 서비스에서는 파일명을 UUID로 변환하거나 특수문자를 제거하는 전처리가 필요합니다.

---

## #5 chunk_id 문자열로 인한 Qdrant 400 에러

### 문제 상황

임베딩 후 Qdrant upsert 시 모든 배치에서 400 에러가 발생했습니다.

```
value 1편_선급등록및검사_0000 is not a valid point ID,
valid values are either an unsigned integer or a UUID
```

### 재현 과정

01번 청킹 단계에서 생성된 JSONL의 chunk_id 구조를 확인했습니다.

```json
{"chunk_id": "1편_선급등록및검사_0000", "text": "...", "source": "..."}
```

이 chunk_id를 그대로 Qdrant point ID로 사용하는 코드를 실행해 에러를 재현했습니다.

### 원인 분석

Qdrant는 포인트 ID로 정수(unsigned integer) 또는 UUID만 허용합니다.
01번 청킹 단계에서 chunk_id를 `파일명_순번` 형태의 문자열로 저장했는데,
이를 그대로 point ID로 사용하면 Qdrant가 거부합니다.

가설 검증 과정:
- 가설 1: JSON 인코딩 문제 → 틀림
- 가설 2: Qdrant ID 형식 제약 → 정답 (에러 메시지에 명시됨)

### 해결 과정

전역 카운터로 정수 ID를 새로 부여하고, 원본 chunk_id는 payload에 보존했습니다.

```python
_point_counter = 0

def make_point(chunk, dense_vec, sparse_weights):
    global _point_counter
    point_id = _point_counter
    _point_counter += 1

    return PointStruct(
        id=point_id,                        # 정수 ID (Qdrant 요구사항)
        payload={
            "chunk_id": chunk["chunk_id"],  # 원본 보존 (파일·순번 추적용)
            "text": text,
            "source": chunk.get("source"),
        }
    )
```

### 검증

수정 후 에러 배치 0개, 6,732건 전체 저장 완료를 확인했습니다.

```
=== 완료 | 저장 6732건 | 에러 배치 0개 ===
컬렉션 포인트 수: 6732
```

### 배운 점

벡터DB마다 ID 형식 제약이 다릅니다. Qdrant는 정수/UUID만 허용하므로 청킹 단계부터 ID 설계를 고려해야 합니다. 원본 식별자는 payload에 보존하면 나중에 어느 파일 몇 번째 청크인지 추적할 수 있습니다. 데이터 설계 시 저장소의 제약 조건을 미리 파악하는 것이 중요합니다.

---

## #6 디스크 100% 초과로 인한 패키지 설치 실패

### 문제 상황

pip install 중 디스크 공간 부족 에러가 발생했습니다.

```
OSError: [Errno 28] 장치에 남은 공간이 없음
```

### 재현 과정

```bash
df -h /home
# /dev/sda2  59G  57G  0  100%
```

59G VMware 고정 할당 디스크가 완전히 꽉 찬 상태였습니다.

### 원인 분석

`du -sh` 명령어로 디렉토리별 용량을 단계적으로 추적했습니다.

```bash
du -sh ~/.cache/pip ~/.cache/huggingface
# 3.6G    /home/es/.cache/pip
# 4.3G    /home/es/.cache/huggingface

du -sh /home/es/*
# 6.1G    /home/es/tooktak         ← ollama-models 비표준 경로 중복 저장
# 5.5G    /home/es/tooktak-backend ← GPU용 torch 포함된 venv
```

torch 기본 설치 시 GPU용 nvidia 라이브러리가 자동으로 딸려왔습니다.

```
nvidia_cudnn_cu13:    366MB
nvidia_cublas:        423MB
triton:               197MB
```

GPU가 없는 VMware 환경에서는 전혀 불필요한 파일들입니다.

### 해결 과정

캐시와 불필요한 파일을 정리해 18G를 확보했습니다.

```bash
pip cache purge                          # pip 캐시 3.6G 제거
rm -rf ~/.cache/huggingface              # HuggingFace 캐시 4.3G 제거
rm -rf ~/tooktak/ollama-models           # 비표준 경로 모델 6G 제거 (ollama pull로 재설치 가능)
rm -rf ~/tooktak-backend/venv            # GPU torch 포함 venv 5.5G 제거
```

이후 CPU 전용 torch로 재설치했습니다.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 검증

```bash
df -h /home
# /dev/sda2  59G  40G  18G  70%
```

18G 확보 후 정상 설치 완료했습니다.

### 배운 점

GPU 없는 환경에서 torch를 설치할 때는 반드시 CPU 전용 인덱스를 지정해야 합니다. 기본 설치 시 불필요한 nvidia 라이브러리가 수GB씩 딸려옵니다. VMware는 디스크가 고정 할당이라 WSL 대비 공간 관리가 더 중요하며, `du -sh`로 디렉토리별 용량을 추적해 원인을 찾는 접근이 효과적이었습니다.
