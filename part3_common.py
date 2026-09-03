"""
Part 3 공용 모듈 — 환경 부트스트랩 + GPU 기반 학습 전략 자동 선택.

사용법 (모든 노트북 첫 셀):
    import part3_common as p3
    p3.bootstrap()                      # 반드시 `import trl` 보다 먼저
    prof = p3.profile("Qwen/Qwen2.5-0.5B-Instruct")
    prof.summary()
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

__all__ = ["bootstrap", "profile", "TrainProfile", "GPUInfo", "free_gpu"]

# ──────────────────────────────────────────────────────────────────────
# 1) 환경 부트스트랩
# ──────────────────────────────────────────────────────────────────────
_BOOTSTRAPPED = False


def bootstrap(quiet: bool = False) -> None:
    """TRL/transformers/deepspeed 조합의 알려진 비호환을 정리한다.

    반드시 `from trl import ...` 이전에 호출할 것.

    고치는 것:
      (1) transformers 5.x 의 `_is_package_available` 은 `return_version` 없이도
          항상 튜플을 돌려준다. TRL 은 bool 을 기대하는 곳과 `[0]` 으로 인덱싱하는
          곳이 섞여 있어, 그대로 두면 비어있지 않은 튜플이 항상 truthy 로 평가되어
          설치되지 않은 선택적 의존성(weave/mergekit/math_verify)을 import 하려다
          죽는다. bool 로도 인덱싱으로도 올바른 값을 주는 타입으로 감싼다.
      (2) vLLM 은 setup.sh 가 더 이상 설치하지 않지만, 누가 따로 깔았을 때를 대비한
          방어막이다. TRL 지원범위(0.10.2~0.11.2) 밖 버전은 모듈 경로가 달라 import 가
          깨지므로, 제거하지 않고 TRL 에게만 '없음'으로 보이게 한다.
      (3) accelerate 가 deepspeed 를 import 할 때 CUDA_HOME 이 없으면 실패한다.
          deepspeed 역시 setup.sh 에서 빠졌지만, 남아 있는 환경을 위해 venv 안의
          nvidia 툴킷 경로를 잡아준다.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    # (3) CUDA_HOME
    if not os.environ.get("CUDA_HOME"):
        import glob
        import importlib.util
        roots = []
        spec = importlib.util.find_spec("nvidia")
        if spec and spec.submodule_search_locations:
            roots += list(spec.submodule_search_locations)
        for root in roots:
            for cand in sorted(glob.glob(os.path.join(root, "cu*"))):
                if os.path.exists(os.path.join(cand, "bin", "nvcc")):
                    os.environ["CUDA_HOME"] = os.path.abspath(cand)
                    break
            if os.environ.get("CUDA_HOME"):
                break

    # (1) + (2) 패키지 탐지 정상화
    import transformers.utils.import_utils as _tiu

    _orig = _tiu._is_package_available

    class _Avail(tuple):
        """(available, version) 튜플이면서 bool 평가도 올바른 반환값."""
        __slots__ = ()

        def __bool__(self):
            return bool(self[0])

    def _patched(pkg_name, return_version=False):
        res = _orig(pkg_name, return_version=True)
        avail, ver = res if isinstance(res, tuple) else (res, "N/A")
        if pkg_name == "vllm":
            avail, ver = False, "N/A"
        return _Avail((avail, ver))

    _tiu._is_package_available = _patched

    # (4) transformers 5.x 는 PreTrainedModel.warnings_issued 를 없앴는데
    #     TRL 의 DPO/GRPO 트레이너가 아직 이 속성에 기록한다. 클래스 레벨로 되살린다.
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "warnings_issued"):
        PreTrainedModel.warnings_issued = {}

    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _BOOTSTRAPPED = True
    if not quiet:
        print(f"✅ bootstrap 완료  (CUDA_HOME={os.environ.get('CUDA_HOME', '미설정')})")


# ──────────────────────────────────────────────────────────────────────
# 2) GPU 탐지
# ──────────────────────────────────────────────────────────────────────
@dataclass
class GPUInfo:
    available: bool
    name: str = "CPU"
    vram_gb: float = 0.0
    bf16: bool = False


def _gpu_info() -> GPUInfo:
    import torch
    if not torch.cuda.is_available():
        return GPUInfo(available=False)
    p = torch.cuda.get_device_properties(0)
    return GPUInfo(
        available=True,
        name=p.name,
        vram_gb=p.total_memory / 1024 ** 3,
        bf16=torch.cuda.is_bf16_supported(),
    )


# ──────────────────────────────────────────────────────────────────────
# 3) 학습 전략 결정
# ──────────────────────────────────────────────────────────────────────
_PARAM_TABLE = {  # 모델 이름 힌트 → 파라미터 수(십억)
    "0.5b": 0.5, "1.5b": 1.5, "1b": 1.0, "2b": 2.0,
    "3b": 3.0, "7b": 7.0, "8b": 8.0, "14b": 14.0,
}


def _params_b(model_name: str) -> float:
    low = model_name.lower()
    for k in sorted(_PARAM_TABLE, key=len, reverse=True):
        if k in low:
            return _PARAM_TABLE[k]
    return 1.5  # 알 수 없으면 보수적으로


@dataclass
class TrainProfile:
    """GPU 여유도에 따라 결정된 학습 전략."""
    gpu: GPUInfo
    model_name: str
    params_b: float
    mode: str                      # "fft" | "lora" | "qlora" | "cpu"
    dtype: str                     # "bfloat16" | "float16" | "float32"
    fft_need_gb: float             # FFT 시 추정 VRAM
    batch_size: int
    grad_accum: int
    max_length: int
    lora: dict = field(default_factory=dict)

    # ---- 편의 메서드 -------------------------------------------------
    @property
    def use_peft(self) -> bool:
        return self.mode in ("lora", "qlora")

    @property
    def torch_dtype(self):
        import torch
        return getattr(torch, self.dtype)

    def lora_config(self):
        """peft.LoraConfig 생성 (FFT 모드면 None)."""
        if not self.use_peft:
            return None
        from peft import LoraConfig
        return LoraConfig(**self.lora)

    def quant_config(self):
        """4bit 양자화 설정 (qlora 모드가 아니면 None)."""
        if self.mode != "qlora":
            return None
        import torch
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if self.gpu.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # 기법별 배치 축소 계수 — 유효배치(batch×accum)는 유지한다.
    #   dpo : policy/ref × chosen/rejected = 로짓 텐서가 SFT 의 약 4배
    #   grpo: 프롬프트당 num_generations 개를 동시에 생성
    _METHOD_DIV = {"sft": 1, "ppo": 1, "grpo": 2, "dpo": 4}

    def trainer_kwargs(self, method: str = "sft") -> dict:
        """*Config(...) 에 그대로 넘길 공통 인자.

        method 를 주면 그 기법의 메모리 특성에 맞춰 배치를 나누고
        gradient_accumulation 을 같은 배수로 늘려 유효배치를 보존한다.
        """
        div = self._METHOD_DIV.get(method, 1)
        bs = max(1, self.batch_size // div)
        ga = self.grad_accum * (self.batch_size // bs)
        kw = dict(
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=ga,
        )
        if self.dtype == "bfloat16":
            kw["bf16"] = True
        elif self.dtype == "float16":
            kw["fp16"] = True
        return kw

    def summary(self) -> None:
        g = self.gpu
        print("=" * 62)
        print("🖥️  실행 환경 분석")
        print("=" * 62)
        if not g.available:
            print("   GPU: 없음 (CPU 모드) — 학습 셀은 매우 느립니다")
        else:
            print(f"   GPU        : {g.name}")
            print(f"   VRAM       : {g.vram_gb:.1f} GB")
            print(f"   bf16 지원  : {'예' if g.bf16 else '아니오'}")
        print(f"   모델       : {self.model_name}  (~{self.params_b}B)")
        print(f"   FFT 추정치 : {self.fft_need_gb:.1f} GB 필요")
        print("-" * 62)
        label = {
            "fft":   "🟢 Full Fine-Tuning (전체 파라미터 학습)",
            "lora":  "🟡 LoRA (어댑터만 학습 — VRAM 부족)",
            "qlora": "🟠 QLoRA (4bit 양자화 + LoRA — VRAM 많이 부족)",
            "cpu":   "⚪ CPU (LoRA, 매우 느림)",
        }[self.mode]
        print(f"   선택된 전략: {label}")
        print(f"   dtype      : {self.dtype}")
        print(f"   배치       : 유효배치 {self.batch_size * self.grad_accum} "
              f"(기법별 자동 조정)")
        for m in ("sft", "ppo", "grpo", "dpo"):
            k = self.trainer_kwargs(m)
            print(f"      - {m:<5}: {k['per_device_train_batch_size']} × accum "
                  f"{k['gradient_accumulation_steps']}")
        print(f"   max_length : {self.max_length}")
        if self.use_peft:
            print(f"   LoRA       : r={self.lora['r']}, alpha={self.lora['lora_alpha']}, "
                  f"target={len(self.lora['target_modules'])}개 모듈")
        print("=" * 62)


def profile(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
            force_mode: str | None = None) -> TrainProfile:
    """GPU 를 분석해 FFT / LoRA / QLoRA 중 하나를 고른다.

    판단 근거는 '이 모델을 FFT 할 때 필요한 VRAM' 추정치이다.
    AdamW 기준 파라미터당 약 16 byte
      = 가중치 bf16 2 + 그래디언트 bf16 2 + Adam m,v fp32 8 + fp32 마스터 4
    여기에 활성값·단편화 여유를 1.6배로 잡는다.

    force_mode 로 "fft"/"lora"/"qlora" 를 직접 지정할 수 있다(강의 시연용).
    """
    g = _gpu_info()
    pb = _params_b(model_name)
    fft_need = pb * 16 * 1.6            # GB

    if not g.available:
        mode, dtype, bs, ga, ml = "cpu", "float32", 1, 8, 256
    elif force_mode:
        mode = force_mode
        dtype = "bfloat16" if g.bf16 else "float16"
        bs, ga, ml = (8, 2, 1024) if mode == "fft" else (4, 4, 1024)
    elif g.vram_gb >= fft_need:
        # A10G/A100 급 — 여유 있으면 전체 파라미터 학습
        mode, dtype = "fft", ("bfloat16" if g.bf16 else "float16")
        bs, ga, ml = 8, 2, 1024
    elif g.vram_gb >= pb * 2 + 6:
        # RTX 4060/4090 급 — 어댑터만 학습
        mode, dtype = "lora", ("bfloat16" if g.bf16 else "float16")
        bs, ga, ml = 4, 4, 1024
    else:
        # VRAM 매우 부족 — 4bit 양자화까지
        mode, dtype = "qlora", ("bfloat16" if g.bf16 else "float16")
        bs, ga, ml = 2, 8, 512

    lora = {}
    if mode in ("lora", "qlora", "cpu"):
        lora = dict(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )

    return TrainProfile(
        gpu=g, model_name=model_name, params_b=pb, mode=mode, dtype=dtype,
        fft_need_gb=fft_need, batch_size=bs, grad_accum=ga,
        max_length=ml, lora=lora,
    )


# ──────────────────────────────────────────────────────────────────────
# 4) 유틸
# ──────────────────────────────────────────────────────────────────────
def free_gpu(*objs) -> None:
    """모델 객체를 지우고 VRAM 을 반환한다."""
    import gc
    import torch
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🧹 VRAM 사용: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
