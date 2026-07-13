# LLM 마스터 5일 과정 (통합)

> **NLP 기초부터 파인튜닝 · 경량화 · 강화학습 정렬 · RAG · Agentic AI까지** — LLM 마스터 특별 교육 시리즈(5개 파트)를 하나의 **5일 집중 과정**으로 통합한 실습 중심 커리큘럼

본 저장소는 **LLM 마스터 과정 특별 교육 시리즈**의 다섯 개 파트(Part 1~5)를 **하루에 한 파트씩, 총 5일 과정**으로 재구성한 통합 실습 자료입니다.
자연어 처리 기초에서 출발해 트랜스포머·파인튜닝(Day 1), 모델 경량화·추론 최적화(Day 2), 강화학습 기반 정렬(Day 3), 지식증강 RAG(Day 4), 바이브 코딩 기반 Agentic AI(Day 5)까지 LLM의 전 주기를 다룹니다.

각 Day의 노트북은 `dN_` 접두사(예: `d1_`, `d2_` …)로 구분되며, 접두사 내부 번호는 원본 파트의 세션 순서를 그대로 따릅니다.

---

## 강의 정보

| 항목 | 내용 |
|---|---|
| **과정명** | (특별교육) [인공지능 : LLM 마스터] — **5일 통합 과정** |
| **구성** | 5일 (Day 1~5, 하루당 한 파트) · 이론 + Jupyter Notebook 실습 병행 |
| **난이도** | 고급 |
| **강사** | **김의중** (아이덴티파이 대표) |
| **교재** | **딥러닝 개념과 활용** (김의중 저) |
| **실습 노트북** | 총 49개 (+ 환경 점검 `setup_check.ipynb`) |

---

## 학습 목표

- 자연어 처리(인코딩·토큰화·임베딩)와 트랜스포머 아키텍처를 이해하고, HuggingFace·LangChain으로 LLM 애플리케이션을 구성한다. **(Day 1)**
- LoRA·PEFT·SFT·Instruction Tuning으로 실제 모델을 파인튜닝하고 LLM-as-a-Judge로 평가한다. **(Day 1)**
- Scaling Law·지식 증류·MoE를 이해하고, 양자화(GPTQ·AWQ·QLoRA·BnB·GGUF)와 Unsloth·vLLM으로 경량화·추론 최적화를 수행한다. **(Day 2)**
- RLHF 파이프라인(PPO·DPO·GRPO)의 원리를 이해하고, 선호도 데이터 기반 DPO·GRPO 정렬 학습을 직접 실행한다. **(Day 3)**
- 벡터 RAG에서 출발해 Advanced·그래프·온톨로지 RAG를 구축하고 RAGAS로 정량 평가한다. **(Day 4)**
- 바이브 코딩과 Claude Code, Tool Calling·MCP·A2A·LangGraph로 멀티 에이전트를 설계하고, 데이터 파이프라인 → 학습 → 평가 → 배포까지 통합 프로젝트를 완성한다. **(Day 5)**

> 💡 각 Day 첫 노트북 `dN_01_nlp_encoding_tokenization.ipynb`는 **공통 인트로(자연어 처리 기초)** 세션입니다. Day 1에서 학습한 뒤 이후 Day에서는 복습용으로 활용하거나 생략할 수 있습니다.

---

## 커리큘럼

### 📅 Day 1 — LLM 아키텍처 분석 및 HuggingFace · LangChain 활용 *(Part 1)*

| # | 세션 | 노트북 |
|---|---|---|
| 00 | 실습 환경 점검 | [setup_check.ipynb](setup_check.ipynb) |
| 01 | 자연어 처리 기초 — 인코딩, 토큰화, 임베딩(Word2Vec) | [d1_01_nlp_encoding_tokenization.ipynb](d1_01_nlp_encoding_tokenization.ipynb) |
| 02 | 트랜스포머 아키텍처 — Attention, BERT, GPT 구조 비교 | [d1_02_transformer_bert_gpt.ipynb](d1_02_transformer_bert_gpt.ipynb) |
| 03 | 생성 AI와 LLM 개요 (sLLM 포함) | [d1_03_llm_overview_sllm.ipynb](d1_03_llm_overview_sllm.ipynb) |
| 04 | Transformers 라이브러리 & HuggingFace Hub | [d1_04_huggingface_ecosystem.ipynb](d1_04_huggingface_ecosystem.ipynb) |
| 05 | LangChain 소개 및 기본 실습 | [d1_05_langchain_practice.ipynb](d1_05_langchain_practice.ipynb) |
| 06 | 파인튜닝 개념 정리 — SFT/CPT/IT, FFT/PEFT | [d1_06_finetuning_overview.ipynb](d1_06_finetuning_overview.ipynb) |
| 07 | HuggingFace Transformers & TRL 기본 사용법 | [d1_07_transformers_trl_basics.ipynb](d1_07_transformers_trl_basics.ipynb) |
| 08 | LoRA vs Full Fine-tuning 이론 비교 | [d1_08_lora_peft_theory.ipynb](d1_08_lora_peft_theory.ipynb) |
| 09 | LoRA vs FFT 실전 비교 실습 | [d1_09_lora_peft_practice.ipynb](d1_09_lora_peft_practice.ipynb) |
| 10 | Next Token Prediction 기반 SFT 실습 | [d1_10_sft_huggingface_trl.ipynb](d1_10_sft_huggingface_trl.ipynb) |
| 11 | Continuous Pretraining — Qwen2.5-1.5B (LoRA) | [d1_11_continuous_learning.ipynb](d1_11_continuous_learning.ipynb) |
| 12 | Instruction Tuning — Qwen2.5-1.5B (LoRA) | [d1_12_instruction_tuning.ipynb](d1_12_instruction_tuning.ipynb) |
| 13 | LLM-as-a-Judge — 자동 평가 메트릭과 GPT-4 평가 | [d1_13_llm_as_judge.ipynb](d1_13_llm_as_judge.ipynb) |

### 📅 Day 2 — 모델 경량화와 추론 최적화 *(Part 2)*

| # | 세션 | 노트북 |
|---|---|---|
| 01 | 자연어 처리 기초 *(공통 인트로)* | [d2_01_nlp_encoding_tokenization.ipynb](d2_01_nlp_encoding_tokenization.ipynb) |
| 02 | 합성 데이터 생성 & Knowledge Distillation | [d2_02_knowledge_distillation.ipynb](d2_02_knowledge_distillation.ipynb) |
| 03 | Scaling Law — 최적의 데이터 규모와 모델 크기 | [d2_03_scaling_law.ipynb](d2_03_scaling_law.ipynb) |
| 04 | MoE (Mixture of Experts) 아키텍처와 DeepSeek | [d2_04_moe_deepseek.ipynb](d2_04_moe_deepseek.ipynb) |
| 05 | DeepSeek R1 Case Study — 학습 파이프라인 분석 | [d2_05_deepseek_r1_analysis.ipynb](d2_05_deepseek_r1_analysis.ipynb) |
| 06 | 양자화 개념 — 정밀도별 비교와 BitsAndBytes 실습 | [d2_06_quantization_concepts.ipynb](d2_06_quantization_concepts.ipynb) |
| 07 | 양자화 기법 비교 — BnB / GGUF / Dynamic Quantization | [d2_07_gptq_awq_qlora.ipynb](d2_07_gptq_awq_qlora.ipynb) |
| 08 | 양자화 실습 — 정밀도별 메모리 비교 | [d2_08_quantization_practice.ipynb](d2_08_quantization_practice.ipynb) |
| 09 | Unsloth를 이용한 고속 QLoRA 파인튜닝 | [d2_09_unsloth_finetuning.ipynb](d2_09_unsloth_finetuning.ipynb) |
| 10 | vLLM 서빙 — PagedAttention과 OpenAI 호환 API | [d2_10_vllm_serving.ipynb](d2_10_vllm_serving.ipynb) |

### 📅 Day 3 — 강화학습을 통한 LLM 정렬 파인튜닝 *(Part 3)*

| # | 세션 | 노트북 |
|---|---|---|
| 01 | 자연어 처리 기초 *(공통 인트로)* | [d3_01_nlp_encoding_tokenization.ipynb](d3_01_nlp_encoding_tokenization.ipynb) |
| 02 | LLM 강화학습 개념 — PPO / DPO / GRPO | [d3_02_rl_basics_alignment.ipynb](d3_02_rl_basics_alignment.ipynb) |
| 03 | Preference 데이터 수집·생성 | [d3_03_preference_data.ipynb](d3_03_preference_data.ipynb) |
| 04 | DPO 학습 실습 — SFT + DPO 파이프라인 | [d3_04_dpo_practice.ipynb](d3_04_dpo_practice.ipynb) |
| 05 | DeepSeek R1 Case Study — 학습 파이프라인 분석 | [d3_05_deepseek_r1_analysis.ipynb](d3_05_deepseek_r1_analysis.ipynb) |
| 06 | Rejection Sampling + SFT 실습 — Best-of-N 데이터 구축 | [d3_06_rejection_sampling_sft.ipynb](d3_06_rejection_sampling_sft.ipynb) |
| 07 | GRPO 실습 — TRL GRPOTrainer로 직접 학습 | [d3_07_grpo_practice.ipynb](d3_07_grpo_practice.ipynb) |

### 📅 Day 4 — 지식증강 RAG (벡터 · 그래프 · 온톨로지) *(Part 4)*

| # | 세션 | 노트북 |
|---|---|---|
| 01 | 자연어 처리 기초 *(공통 인트로)* | [d4_01_nlp_encoding_tokenization.ipynb](d4_01_nlp_encoding_tokenization.ipynb) |
| 02 | RAG 파이프라인 — ChromaDB & 시맨틱 서치 | [d4_02_rag_fundamentals.ipynb](d4_02_rag_fundamentals.ipynb) |
| 03 | 벡터 DB 심층 비교 분석 및 실습 | [d4_03_vector_db_comparison.ipynb](d4_03_vector_db_comparison.ipynb) |
| 04 | LangChain RAG 어플리케이션 구현 (PDF 챗봇) | [d4_04_rag_practice.ipynb](d4_04_rag_practice.ipynb) |
| 05 | Advanced RAG — HyDE, Reranking, Ensemble Retriever | [d4_05_advanced_rag_base.ipynb](d4_05_advanced_rag_base.ipynb) |
| 06 | 그래프 RAG — 지식 그래프 기반 검색 증강 | [d4_06_graph_rag.ipynb](d4_06_graph_rag.ipynb) |
| 07 | 온톨로지 RAG — RDF · OWL 추론 (rdflib) | [d4_07_ontology_rag.ipynb](d4_07_ontology_rag.ipynb) |
| 08 | RAG 평가 — RAGAS & LLM-as-a-Judge | [d4_08_rag_evaluation.ipynb](d4_08_rag_evaluation.ipynb) |

### 📅 Day 5 — 바이브 코딩을 이용한 Agentic AI와 Harness 설계 *(Part 5)*

| # | 세션 | 노트북 |
|---|---|---|
| 01 | 자연어 처리 기초 *(공통 인트로)* | [d5_01_nlp_encoding_tokenization.ipynb](d5_01_nlp_encoding_tokenization.ipynb) |
| 02 | 바이브 코딩(Vibe Coding)이란? | [d5_02_vibe_coding_intro.ipynb](d5_02_vibe_coding_intro.ipynb) |
| 03 | Claude Code를 이용한 AI Agent 구현 실습 | [d5_03_claude_code_agent.ipynb](d5_03_claude_code_agent.ipynb) |
| 04 | Tool Calling (Function Calling) 개념 | [d5_04_tool_calling_function.ipynb](d5_04_tool_calling_function.ipynb) |
| 05 | MCP(Model Context Protocol) 기반 에이전트 구현 | [d5_05_mcp_agent.ipynb](d5_05_mcp_agent.ipynb) |
| 06 | A2A(Agent-to-Agent) 프로토콜 기반 멀티 에이전트 | [d5_06_a2a_protocol.ipynb](d5_06_a2a_protocol.ipynb) |
| 07 | Agent AI 기술 스택과 LangGraph 멀티 에이전트 워크플로우 | [d5_07_agent_tech_stack_langgraph.ipynb](d5_07_agent_tech_stack_langgraph.ipynb) |
| 08 | 프로젝트 데이터 파이프라인 구축 | [d5_08_data_pipeline_training.ipynb](d5_08_data_pipeline_training.ipynb) |
| 09 | 프로젝트 학습 — MCP + LangGraph + A2A 통합 에이전트 | [d5_09_project_training.ipynb](d5_09_project_training.ipynb) |
| 10 | 프로젝트 성능 평가 및 반복 개선 | [d5_10_evaluation.ipynb](d5_10_evaluation.ipynb) |
| 11 | 프로젝트 배포 — FastAPI + Streamlit | [d5_11_deployment.ipynb](d5_11_deployment.ipynb) |

---

## 실습 환경 설정

깡통(clean) Ubuntu 상태에서 다음 스크립트로 실습 환경을 자동 구성합니다. `sudo` 없이 `uv`로 독립형 **Python 3.11**을 설치하고 가상환경을 생성합니다.

```bash
bash setup.sh
```

생성되는 가상환경:

| 환경 | 용도 |
|---|---|
| `venv` | 메인 환경 (torch 2.11 / transformers 4.57 / vllm / trl / peft / langchain 0.3 / gensim / chromadb / langgraph / fastapi / streamlit 등) — 대부분의 노트북에서 사용 |
| `venv-quant` | 양자화 전용 (torch 2.2 / transformers 4.46 / auto-gptq / autoawq) — Day 2 일부 양자화 데모에서만 사용 |

> Python 3.11을 고정하는 이유: 최신 Python(3.14)에서는 gensim·auto-gptq·autoawq 등 다수 ML 패키지의 사전 빌드 휠이 없어 설치가 깨지기 때문입니다.
> auto-gptq·autoawq는 구버전 torch(2.2)·transformers를 요구해 메인 환경(torch 2.11)과 공존이 불가하므로 별도 `venv-quant`로 분리합니다.

### Day별 실행 요건

| Day | GPU 필요 여부 | 필요한 외부 서비스 / API 키 |
|---|---|---|
| Day 1 | 파인튜닝 세션(09~12)은 GPU 권장 (소형 Qwen2.5-1.5B + LoRA) | LLM-as-a-Judge(13)에 `OPENAI_API_KEY` |
| Day 2 | 양자화·Unsloth·vLLM 세션은 GPU 필요 | — |
| Day 3 | DPO(04)·GRPO(07)는 GPU 필수 (4bit + LoRA, RTX 4060 8GB↑ 권장) | — |
| Day 4 | 대부분 CPU 가능 | 그래프 RAG(06)는 **Neo4j**(Docker/Aura), 생성·평가는 `OPENAI_API_KEY` |
| Day 5 | 대부분 API 기반, GPU 불필요 | `OPENAI_API_KEY`, Claude Code/바이브 코딩(03)에 `ANTHROPIC_API_KEY` |

> API 키는 노트북과 같은 폴더의 `.env` 파일에 넣어두면 자동으로 로드됩니다.

### 설치 확인

환경 구성 후 [setup_check.ipynb](setup_check.ipynb)를 실행해 주요 패키지와 GPU/CUDA 인식 여부를 점검하세요.

```bash
source venv/bin/activate
jupyter notebook   # 또는 VS Code / JupyterLab 사용
```

---

## 원본 시리즈 구성

본 5일 과정은 아래 **LLM 마스터 과정 특별 교육 시리즈(5개 파트)** 를 통합한 것입니다. 각 파트는 원래 2일 / 12시간 과정입니다. (난이도: 고급)

| Day | 파트 | 일정 | 주제 |
|---|---|---|---|
| Day 1 | Part 1 | 2026-08-08 ~ 08-09 | LLM 아키텍처 분석 및 HuggingFace, LangChain 활용 |
| Day 2 | Part 2 | 2026-08-15 ~ 08-16 | 모델 경량화와 추론 최적화 |
| Day 3 | Part 3 | 2026-08-22 ~ 08-23 | 강화학습을 통한 LLM 정렬 파인튜닝 |
| Day 4 | Part 4 | 2026-09-12 ~ 09-13 | 지식증강 — 벡터 RAG & 그래프 RAG & 온톨로지 RAG |
| Day 5 | Part 5 | 2026-09-19 ~ 09-20 | 바이브 코딩을 이용한 Agentic AI와 Harness 설계 |

---

## 라이선스 및 저작권

본 교육 자료의 모든 노트북과 코드는 **© AIDENTIFY. All rights reserved.**
교육 목적 외 무단 복제·배포를 금합니다.
