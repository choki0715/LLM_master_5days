#!/bin/bash
# ===========================================
# LLM 파인튜닝 실습 환경 자동 설정 스크립트
# 대상: 깡통 Ubuntu (아무것도 없는 상태)
# 사용법: bash setup.sh
#
# 구조 (Python 3.11 고정):
#   venv        — 메인 환경 (torch 2.11 / transformers 4.57 / vllm / langchain / gensim ...)
#   venv-quant  — 양자화 전용 (torch 2.2 / transformers 4.46 / auto-gptq / autoawq)
#                 ※ auto-gptq(torch 2.2)·autoawq(구 transformers)는 메인 스택(vllm=torch 2.11)과
#                    절대 공존 불가라 분리. 22·25강 양자화 데모에서만 venv-quant 커널 사용.
#
# Python 3.11 을 쓰는 이유: Ubuntu 26.04 기본 Python 3.14 에서는 gensim·auto-gptq·autoawq 등
# 다수 ML 패키지의 사전 빌드 휠이 없어 설치가 깨짐. 3.11 은 휠이 모두 제공되는 안정 버전.
# Ubuntu 26.04 apt 에는 python3.11 패키지가 없으므로 uv(sudo 불필요)로 독립형 3.11 을 설치.
# ===========================================

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
NC='\033[0m'

TOTAL_STEPS=13
CURRENT_STEP=0

print_step() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    local pct=$((CURRENT_STEP * 100 / TOTAL_STEPS))
    local filled=$((pct / 5))
    local empty=$((20 - filled))
    local bar=""
    if [ "$filled" -gt 0 ]; then
        bar=$(printf '%0.s#' $(seq 1 "$filled"))
    fi
    if [ "$empty" -gt 0 ]; then
        bar="${bar}$(printf '%0.s-' $(seq 1 "$empty"))"
    fi
    echo ""
    echo -e "${CYAN}[${CURRENT_STEP}/${TOTAL_STEPS}]${NC} ${GREEN}$1${NC}"
    echo -e "${BLUE}  [${bar}] ${pct}%${NC}"
}
print_warn() { echo -e "  ${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "  ${RED}[ERROR]${NC} $1"; }
print_ok() { echo -e "  ${GREEN}[OK]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo -e "${CYAN}==========================================${NC}"
echo -e "${CYAN}   LLM 파인튜닝 실습 환경 설정${NC}"
echo -e "${CYAN}   Python 3.11 / venv + venv-quant${NC}"
echo -e "${CYAN}   총 ${TOTAL_STEPS}단계를 진행합니다${NC}"
echo -e "${CYAN}==========================================${NC}"

# ----- 1. 시스템 패키지 설치 -----
print_step "시스템 패키지 설치 (sudo 필요)"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential cmake \
    libaio-dev \
    curl wget git \
    > /dev/null 2>&1
# 한글 폰트는 배포판/버전마다 패키지명이 달라 실패해도 setup을 중단하지 않음 (best-effort)
# (예: Ubuntu 26.04에서 fonts-nanum-coding 제거됨)
sudo apt-get install -y -qq fonts-nanum fonts-nanum-extra fonts-noto-cjk > /dev/null 2>&1 \
    || print_warn "일부 한글 폰트 설치 실패 — 그래프 한글이 깨질 수 있습니다 (수동: sudo apt install fonts-nanum)"
fc-cache -fv > /dev/null 2>&1 || true
rm -rf ~/.cache/matplotlib 2>/dev/null || true
print_ok "시스템 패키지 설치 완료 (Nanum + Noto CJK 한글 폰트 포함, matplotlib 캐시 초기화)"

# ----- 2. NVIDIA 드라이버 확인 -----
print_step "NVIDIA 드라이버 확인"
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    print_ok "드라이버 ${DRIVER_VER} / GPU: ${GPU_NAME}"
else
    print_warn "NVIDIA 드라이버가 없습니다. GPU 학습이 불가능합니다."
    print_warn "설치: sudo apt install nvidia-driver-550  (이후 재부팅 필요)"
fi

# ----- 3. uv + Python 3.11 설치 -----
print_step "uv + Python 3.11 설치 (sudo 불필요)"
if ! command -v uv &> /dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
"$UV" python install 3.11 > /dev/null 2>&1
PY311="$("$UV" python find 3.11)"
print_ok "uv $("$UV" --version | awk '{print $2}') / Python 3.11 ($PY311)"

# ----- 4. 메인 가상환경 생성 (venv, 3.11) -----
print_step "메인 가상환경 생성 (venv, Python 3.11)"
if [ -d "venv" ]; then
    print_warn "venv가 이미 존재합니다. 기존 환경을 사용합니다."
else
    "$UV" venv --python 3.11 venv > /dev/null 2>&1
    print_ok "venv 생성 완료 ($(venv/bin/python --version))"
fi
MAIN="$SCRIPT_DIR/venv/bin/python"
mpip() { "$UV" pip install --python "$MAIN" "$@"; }

# ----- 5. 메인 스택 설치 (PyTorch + ML + RAG + 서빙) -----
print_step "메인 스택 설치 - 시간이 걸립니다 (torch/vllm 등 대용량)"
# torch 는 vllm 이 2.11 을 요구하므로 처음부터 2.11 로 맞춤 (이후 vllm 이 덮어쓰지 않게)
mpip pip setuptools wheel -q
mpip torch==2.11.0 torchvision==0.26.0 torchaudio --index-url https://download.pytorch.org/whl/cu128 -q

# 핵심 ML
mpip "transformers==4.57.2" "accelerate==1.13.0" "datasets==4.8.4" \
    "peft==0.18.1" "trl==0.23.0" "bitsandbytes==0.49.2" "huggingface-hub==0.36.2" -q

# LangChain / RAG  (langgraph 1.0+ 는 langchain-core>=1.4 를 요구해 langchain 0.3.x 와 충돌 → <1.0 캡)
mpip "langchain==0.3.28" "langchain-openai==0.3.35" "langchain-community==0.3.31" \
    "langchain-ollama==0.3.10" "langchain-chroma==0.2.6" "langchain-experimental>=0.3.0" \
    "chromadb==1.5.7" "sentence-transformers==5.4.0" "rank_bm25==0.2.2" -q

# OpenAI / Anthropic / 데이터 / 토큰화  (gensim 은 3.11 에선 사전 빌드 휠 제공 → 그냥 설치됨)
mpip "openai==2.31.0" "anthropic>=0.40.0" "tiktoken==0.12.0" "pandas==3.0.2" \
    "numpy>=2.3.3" "python-dotenv==1.2.2" "tqdm==4.67.3" "pypdf>=4.0.0" "gensim>=4.3.0" -q

# 평가
mpip "ragas" "nltk==3.9.4" "rouge-score==0.1.2" "bert-score==0.3.13" \
    "evaluate>=0.4.0" "scikit-learn>=1.3.0" -q

# 시각화 / UI / 그래프
mpip "matplotlib==3.10.8" "streamlit==1.56.0" "networkx>=3.0" -q

# API 서버 / HTTP
mpip "fastapi>=0.110.0" "uvicorn[standard]>=0.27.0" "pydantic>=2.0.0" \
    "requests>=2.31.0" "httpx>=0.27.0" -q

# 서빙 / 고급 (실패 허용)
mpip "vllm" -q                  || print_warn "vllm 설치 실패 — 서빙(vLLM) 데모만 영향"
mpip "deepspeed==0.18.9" -q     || print_warn "deepspeed 설치 실패 — DeepSpeed 데모만 영향"
mpip "llama-cpp-python==0.3.20" -q || print_warn "llama-cpp-python 설치 실패 — GGUF 로컬 추론 데모만 영향"
mpip "unsloth" --no-deps -q     || print_warn "Unsloth 설치 실패 — 21 노트북만 영향 (수동: uv pip install unsloth)"

# Agent / MCP / LangGraph (langgraph<1.0)  +  Graph DB / KG
mpip "mcp[cli]>=1.2.0" "langgraph>=0.2.0,<1.0" -q
mpip "neo4j>=5.0.0" "psycopg2-binary>=2.9.0" "rdflib>=7.0.0" "pyoxigraph>=0.5.0" "faiss-cpu>=1.7.4" -q
print_ok "메인 스택 설치 완료"

# ----- 6. 양자화 전용 환경 (venv-quant, 3.11) -----
print_step "양자화 전용 환경 생성 + 설치 (venv-quant) — auto-gptq / autoawq"
# 이 둘은 구버전 torch(2.2)·구버전 transformers 를 요구해 메인(torch 2.11)과 공존 불가 → 별도 venv.
# 22·25강 양자화 데모를 돌릴 때만 Jupyter 커널 "Python (LLM Quant 3.11)" 선택.
if [ -d "venv-quant" ]; then
    print_warn "venv-quant가 이미 존재합니다. 기존 환경을 사용합니다."
else
    "$UV" venv --python 3.11 venv-quant > /dev/null 2>&1
fi
QPY="$SCRIPT_DIR/venv-quant/bin/python"
qpip() { "$UV" pip install --python "$QPY" "$@"; }
# 각 단계마다 torch==2.2.2 를 함께 고정 — autoawq 등이 상한 없는 torch 요구로 최신 torch 를 끌어오는 것 차단
qpip torch==2.2.2 torchvision==0.17.2 triton==2.2.0 "numpy<2" \
    --index-url https://download.pytorch.org/whl/cu121 -q \
    && {
        # shard_checkpoint(autoawq) + PytorchGELUTanh(autoawq) 둘 다 있는 마지막 라인 = transformers 4.46
        # PEFT_TYPE_TO_MODEL_MAPPING(auto-gptq) 가 있는 peft = 0.10.0
        qpip "transformers==4.46.3" "tokenizers==0.20.3" "accelerate==0.34.2" \
            "optimum==1.23.3" "peft==0.10.0" "datasets" "sentencepiece" "protobuf" "torch==2.2.2" -q
        qpip "auto-gptq==0.7.1" \
            --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/ "torch==2.2.2" -q \
            || print_warn "auto-gptq 설치 실패 — 22강 GPTQ 데모만 영향"
        qpip "autoawq==0.2.5" "autoawq-kernels==0.0.6" "torch==2.2.2" -q \
            || print_warn "autoawq 설치 실패 — 22·25강 AWQ 데모만 영향"
        print_ok "venv-quant 설치 완료 (torch 2.2.2 / transformers 4.46.3 / auto-gptq / autoawq)"
    } || print_warn "venv-quant torch 설치 실패 — 양자화 환경을 건너뜁니다"

# ----- 7. Jupyter 커널 등록 (두 환경) -----
print_step "Jupyter 커널 등록 (venv + venv-quant)"
mpip ipykernel -q
"$MAIN" -m ipykernel install --user --name venv --display-name "Python (LLM 3.11)" > /dev/null 2>&1
print_ok "커널 'Python (LLM 3.11)' 등록"
if [ -x "$QPY" ]; then
    qpip ipykernel -q 2>/dev/null || true
    "$QPY" -m ipykernel install --user --name venv-quant --display-name "Python (LLM Quant 3.11)" > /dev/null 2>&1 \
        && print_ok "커널 'Python (LLM Quant 3.11)' 등록"
fi

# ----- 8. Ollama 설치 -----
print_step "Ollama 설치"
if command -v ollama &> /dev/null; then
    print_warn "Ollama가 이미 설치되어 있습니다: $(ollama --version 2>&1)"
else
    curl -fsSL https://ollama.ai/install.sh | sh
    print_ok "Ollama 설치 완료"
fi

# ----- 9. .env 파일 생성 -----
print_step ".env 파일 확인"
if [ -f ".env" ]; then
    print_warn ".env 파일이 이미 존재합니다. 건너뜁니다."
else
    cp .env.example .env
    print_ok ".env 파일 생성 완료 — API 키를 입력해주세요"
fi

# ----- 10. GPU 점검 -----
print_step "GPU 점검 (메인 venv)"
"$MAIN" -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
    print(f'  CUDA: {torch.version.cuda}')
else:
    print('  GPU를 찾을 수 없습니다!')
"
print_ok "GPU 점검 완료"

# ----- 11. 설치 확인 -----
print_step "설치된 주요 패키지 확인"
echo "  --- 메인 venv ---"
"$MAIN" -c "
import importlib
pkgs = ['torch','transformers','peft','trl','datasets','accelerate','bitsandbytes',
    'langchain','langchain_openai','langchain_community','langchain_experimental',
    'chromadb','openai','anthropic','tiktoken','sentence_transformers','gensim',
    'networkx','neo4j','psycopg2','rdflib','pyoxigraph','faiss','mcp','langgraph','httpx','fastapi',
    'pydantic','streamlit','vllm','pypdf']
for p in pkgs:
    try:
        m = importlib.import_module(p)
        print(f'  {p:25s} {getattr(m,\"__version__\",\"OK\")}')
    except ImportError:
        print(f'  {p:25s} NOT INSTALLED')
"
if [ -x "$QPY" ]; then
    echo "  --- venv-quant (양자화 전용) ---"
    "$QPY" -c "
import warnings; warnings.filterwarnings('ignore')
import importlib
for p in ['torch','transformers','auto_gptq','awq']:
    try:
        m = importlib.import_module(p)
        print(f'  {p:25s} {getattr(m,\"__version__\",\"OK\")}')
    except ImportError:
        print(f'  {p:25s} NOT INSTALLED')
" 2>/dev/null
fi
print_ok "패키지 확인 완료"

# ----- 12. VS Code 설치 -----
# 데스크톱 VS Code(apt 패키지 'code')만 검사한다. `command -v code` 는 VS Code Remote 접속 시
# ~/.vscode-server 의 서버 CLI 까지 잡아 "이미 설치됨"으로 오판 → 데스크톱 설치를 건너뛰는 버그.
print_step "VS Code 설치"
if dpkg -s code &> /dev/null; then
    print_ok "VS Code(데스크톱)가 이미 설치되어 있습니다."
else
    print_ok "VS Code 설치 중..."
    wget -qO /tmp/vscode.deb "https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64"
    sudo apt-get install -y -qq /tmp/vscode.deb > /dev/null 2>&1 \
        || { sudo dpkg -i /tmp/vscode.deb; sudo apt-get install -f -y -qq > /dev/null 2>&1; }
    rm -f /tmp/vscode.deb
    if dpkg -s code &> /dev/null; then print_ok "VS Code 설치 완료"; else print_warn "VS Code 설치 실패 — 수동: https://code.visualstudio.com/download"; fi
fi

# ----- 13. 완료 안내 -----
print_step "설정 완료"
echo ""
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  [####################] 100% 설정 완료!${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""
echo "  다음 단계:"
echo "  1. .env 파일에 API 키 입력 (OPENAI_API_KEY=sk-... / HF_TOKEN=hf_...)"
echo "  2. VS Code에서 커널 선택:"
echo "     - 일반 노트북          → 'Python (LLM 3.11)'   (venv)"
echo "     - 22·25강 양자화 데모  → 'Python (LLM Quant 3.11)' (venv-quant)"
echo "  3. setup_check.ipynb 실행하여 최종 점검"
echo ""
