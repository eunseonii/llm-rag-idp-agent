# LLM, RAG 기반 지능형 문서처리(IDP) 에이전트 개발

> 팀 프로젝트 뚝딱(TookTak)의 RAG 파이프라인 담당 파트입니다.

## 담당 역할
4인 팀 프로젝트에서 **RAG 파이프라인 전담 개발**
- 문서 청킹 및 전처리
- BGE-M3 임베딩 + Qdrant 벡터 저장소 구축
- LangChain 기반 RAG 파이프라인 설계 및 구현
- Ollama + Qwen3:8b 로컬 LLM 연동
- FastAPI 연동용 generate_sections() 인터페이스 설계

## 기술 스택
| 구분 | 기술 |
|------|------|
| 임베딩 | BGE-M3 |
| 벡터DB | Qdrant (Docker) |
| RAG 프레임워크 | LangChain |
| LLM | Ollama + Qwen3:8b |
| PDF 처리 | PyMuPDF |
| 백엔드 | FastAPI |
| 프론트엔드 | React 18 |

## 파이프라인 구조
PDF 코퍼스 → 청킹 → BGE-M3 임베딩 → Qdrant 저장
                                          ↓
사용자 입력(컨셉) → 쿼리 생성 → 유사도 검색 → Qwen3:8b → 섹션별 텍스트 생성

## 파일 구조
- 01_load_and_chunk.py       : PDF 로딩 및 청킹
- 02_embed_and_store.py      : BGE-M3 임베딩 + Qdrant 저장
- 03_rag_pipeline.py         : RAG + LLM 텍스트 생성 (FastAPI 연동용)
- backend/app/rag/           : FastAPI 연동 RAG 모듈
- docs/TROUBLESHOOTING.md    : 트러블슈팅 기록
- docs/development-log.md    : 개발 과정 기록

## 코퍼스 구성
- 실제 프로젝트 기획서 PDF 3개
- LLM 생성 문서 제외 (환각 증폭 방지)
- SPRI AI 산업 동향 보고서 2개 (2024~2025)
- 총 1305개 청크
- 현재 IT/AI 도메인 중심 구성, 추후 도메인 확장 예정

## 개발 현황
- [x] PDF 코퍼스 구성 및 청킹 (121개 청크)
- [x] BGE-M3 임베딩 + Qdrant 인덱싱
- [x] RAG + LLM 텍스트 생성 동작 확인 (섹션별 텍스트 생성)
- [ ] FastAPI 연동
- [ ] 실제 PDF 양식 필드 매핑 검증
- [ ] 프롬프트 최적화 (수치 환각 억제)
