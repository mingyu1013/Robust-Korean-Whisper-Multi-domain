# -*- coding: utf-8 -*-
"""
dialect_mas_lora_to_json.py

- 입력 JSON을 읽어서
- Tel 도메인은 전부 필터링하고
- 오디오를 로드해서 Whisper+MAS-LoRA 모델로 디코딩 후
- tel_off_lora_to_json.py와 동일한 JSON 형식으로 저장.

게이트 정책:
- domain == "<Tel>"         → 애초에 평가 대상에서 제거
- domain in DIALECT(5개)    → 5 expert 균등 가중치 (1/5) mixture
- domain == "<Day>"         → 위와 동일하게 5 expert 균등 가중치 mixture
- 그 외(<News> 등)         → gate = 0 → Stage1 단일 LoRA 경로만 사용
"""

import os, json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ── 설정 (필요한 부분만 수정해서 사용) ───────────────────────────────────────

JSON_PATH   = "/home/work/cymg0001/preprocessed_audio/test/merged_test4.json"  # ★ 평가용 JSON 경로
AUDIO_ROOT  = "/home/work/cymg0001/preprocessed_audio/test"          # ★ 오디오 root 디렉터리
PATH_TEMPLATE = "{id}.wav"   # 예: "{domain}/{id}.wav" 형태면 여기 수정

OUTPUT_DIR  = "/home/work/cymg0001/preprocessed_audio/mg/last_model_eval_outputs"
RUN_NAME    = "mas_lora_small_dialect_from_single_eval"

BASE_MODEL_NAME = "openai/whisper-small"

# Stage2 MAS-LoRA 학습 결과 디렉터리 (dialect_mas_lora.py에서 저장한 best)
MAS_MODEL_DIR   = "/home/work/cymg0001/preprocessed_audio/mg/dialect_mas_lora_output/mas_lora_small_dialect_from_single/best"
MAS_STATE_PATH  = os.path.join(MAS_MODEL_DIR, "pytorch_mas.bin")
PROCESSOR_DIR   = os.path.join(MAS_MODEL_DIR, "processor")

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE      = 4
NUM_WORKERS     = 0
MAX_NEW_TOKENS  = 128
LANGUAGE        = "ko"
TASK            = "transcribe"   # 또는 "translate"
TARGET_SR       = 16000

EXCLUDE_TEL     = True  # Tel 도메인은 반드시 제외

# MAS-LoRA 설정 (학습 때와 동일하게 맞춰야 함)
ACTIVE_DIALECTS = [
    "<Dia><JL>",
    "<Dia><GW>",
    "<Dia><GS>",
    "<Dia><JJ>",
    "<Dia><CC>",
]
N_EXPERTS       = len(ACTIVE_DIALECTS)
MAS_R           = 8
MAS_ALPHA       = 16.0
MAS_DROPOUT     = 0.1
TARGETS         = ("q_proj", "v_proj")


# ── 유틸: JSON 로드 / Tel 필터 ────────────────────────────────────────────────

def load_items(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()
    if not raw:
        return []
    if raw[0] == '[':
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]

def is_tel_item(rec: Dict[str, Any]) -> bool:
    """
    tel_off_lora_to_json.py의 Tel 필터 로직과 동일하게 유지
    - id prefix 'tel_'
    - domain == "<Tel>"
    - domain_id / domain_idx / domain_label == 1
    """
    _id = str(rec.get("id", ""))
    dom = rec.get("domain")
    did = rec.get("domain_id") or rec.get("domain_idx") or rec.get("domain_label")
    if isinstance(_id, str) and _id.lower().startswith("tel_"):
        return True
    if isinstance(dom, str) and dom == "<Tel>":
        return True
    if isinstance(did, int) and did == 1:
        return True
    return False


# ── 오디오 경로 생성기 (원본 코드와 동일 구조) ───────────────────────────────

AUDIO_EXTS = [".wav", ".flac", ".mp3", ".m4a"]

def resolve_audio_path(audio_root: str, _id: str, domain: Optional[str],
                       path_template: Optional[str]) -> Optional[str]:
    candidates = []
    if path_template:
        has_ext = any(path_template.endswith(ext) for ext in AUDIO_EXTS)
        if has_ext:
            rel = path_template.format(id=_id, domain=(domain or ""))
            candidates = [os.path.join(audio_root, rel)]
        else:
            for ext in AUDIO_EXTS:
                rel = (path_template + ext).format(id=_id, domain=(domain or ""))
                candidates.append(os.path.join(audio_root, rel))
    else:
        for ext in AUDIO_EXTS:
            candidates.append(os.path.join(audio_root, f"{_id}{ext}"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


# ── Dataset / Collator (원본과 동일한 형식 유지) ─────────────────────────────

class IdOnlyDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], audio_root: str, path_template: Optional[str]):
        self.items = items
        self.audio_root = audio_root
        self.path_template = path_template

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        _id = rec.get("id")
        domain = rec.get("domain")
        ref = rec.get("text")
        if _id is None:
            raise ValueError(f"[IDX {idx}] 'id'가 없습니다: keys={list(rec.keys())}")
        audio_path = resolve_audio_path(self.audio_root, str(_id), domain, self.path_template)
        return {"id": _id, "domain": domain, "ref": ref, "audio_path": audio_path}

@dataclass
class Collator:
    target_sr: int
    def __call__(self, batch: List[Dict[str, Any]]):
        kept, dropped = [], []
        for b in batch:
            if b["audio_path"] and os.path.isfile(b["audio_path"]):
                kept.append(b)
            else:
                dropped.append(b)
        if len(kept) == 0:
            return {"waves": [], "metas": [], "dropped": dropped}

        waves, metas = [], []
        for b in kept:
            wav, sr = torchaudio.load(b["audio_path"])
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0)
            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            waves.append(wav.cpu())
            metas.append({
                "id": b["id"],
                "domain": b["domain"],
                "ref": b["ref"],
                "audio_path": b["audio_path"],
            })
        return {"waves": waves, "metas": metas, "dropped": dropped}


# ── MAS-LoRA 모듈 정의 (학습 코드와 동일) ────────────────────────────────────

class MASLoRALinear(torch.nn.Module):
    """
    - base: Stage1 단일 LoRA까지 merge된 nn.Linear (freeze 가정)
    - As/Bs: 각 expert 별 LoRA 파라미터 (trainable)
    - _expert_weights: (B, n_experts) — 외부에서 설정하는 gate (one-hot / uniform 등)
    """
    def __init__(self, base: torch.nn.Linear, r: int, alpha: float,
                 dropout: float, n_experts: int):
        super().__init__()
        assert isinstance(base, torch.nn.Linear), "MASLoRALinear base는 nn.Linear 여야 합니다."
        self.base = base
        self.n_experts = int(n_experts)
        self.r = int(r)
        self.scaling = alpha / max(1, self.r)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

        in_f  = base.in_features
        out_f = base.out_features
        self.in_features  = in_f
        self.out_features = out_f

        self.As = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(self.r, in_f))
            for _ in range(self.n_experts)
        ])
        self.Bs = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(out_f, self.r))
            for _ in range(self.n_experts)
        ])
        self.reset_parameters()

        self._expert_weights: Optional[torch.Tensor] = None

    def reset_parameters(self):
        for A, B in zip(self.As, self.Bs):
            torch.nn.init.kaiming_uniform_(A, a=5 ** 0.5)
            torch.nn.init.zeros_(B)

    @torch.no_grad()
    def set_expert_weights(self, w: Optional[torch.Tensor]):
        """
        w: (B, n_experts) or None
        """
        self._expert_weights = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base path (Stage1 단일 LoRA merge된 Linear)
        out = self.base(x)

        if self.r == 0 or self._expert_weights is None:
            return out

        w = self._expert_weights
        dev = x.device
        if w.device != dev:
            w = w.to(dev)

        # expert 파라미터도 device 동기화
        for i in range(self.n_experts):
            if self.As[i].device != dev:
                self.As[i].data = self.As[i].data.to(dev)
                self.Bs[i].data = self.Bs[i].data.to(dev)

        # x: (B, T, C) 또는 (B, C)
        if x.dim() == 3:
            Bsz, T, C = x.shape
            x2 = x.reshape(Bsz * T, C)
            # w: (B, E) → (B*T, E)
            if w.dim() == 2 and w.shape[0] == Bsz:
                w2 = w.repeat_interleave(T, dim=0)
            else:
                w2 = w
        else:
            Bsz, C = x.shape
            x2 = x
            w2 = w  # (B, E)

        lora_sum = None
        for e in range(self.n_experts):
            A = self.As[e]      # (r, in_f)
            Bmat = self.Bs[e]   # (out_f, r)
            l = x2 @ A.t()      # (B*, r)
            l = self.dropout(l)
            l = l @ Bmat.t()    # (B*, out_f)
            we = w2[:, e].unsqueeze(1)  # (B*, 1)
            l = l * we
            lora_sum = l if lora_sum is None else (lora_sum + l)

        if x.dim() == 3:
            lora_sum = lora_sum.view(Bsz, T, self.out_features)

        return out + lora_sum * self.scaling


def inject_mas_encoder(model: torch.nn.Module,
                       targets=("q_proj", "v_proj"),
                       r=8, alpha=16.0, dropout=0.1, n_experts=5):
    """
    Encoder self-attn q_proj/v_proj 자리에 MASLoRALinear 주입.
    base는 현재 Linear를 그대로 사용.
    """
    replaced = 0
    for name, module in model.named_modules():
        # encoder self-attn만
        if ".encoder.layers." not in name:
            continue
        if ".self_attn." not in name:
            continue
        if not any(name.endswith(f".{t}") for t in targets):
            continue

        parent_name = name.rsplit(".", 1)[0]
        attr = name.split(".")[-1]
        parent = model.get_submodule(parent_name)
        sub = getattr(parent, attr, None)
        if not isinstance(sub, torch.nn.Linear):
            continue

        mas = MASLoRALinear(sub, r=r, alpha=alpha, dropout=dropout, n_experts=n_experts)
        setattr(parent, attr, mas)
        replaced += 1

    print(f"[MAS-ENC] injected {replaced} modules on encoder.self targets={targets}")


def set_all_mas_weights(model: torch.nn.Module, w: Optional[torch.Tensor]):
    """
    모델 안의 모든 MASLoRALinear 모듈에 동일한 gate w(B, E)를 설정
    """
    for m in model.modules():
        if isinstance(m, MASLoRALinear):
            m.set_expert_weights(w)


# ── MAS 모델 로드 ────────────────────────────────────────────────────────────

def load_mas_model(base_model_name: str,
                   mas_dir: str,
                   state_path: str,
                   device: str):
    """
    - base Whisper-small 로드
    - encoder q/v에 MASLoRALinear 주입
    - 학습 시 저장한 state_dict(pytorch_mas.bin) 로드
    - processor는 MAS 모델 디렉터리에 저장된 것 사용
    """
    # Processor
    if os.path.isdir(PROCESSOR_DIR):
        print(f"[LOAD] processor from {PROCESSOR_DIR}")
        processor = WhisperProcessor.from_pretrained(PROCESSOR_DIR)
    else:
        print(f"[LOAD] processor from base {base_model_name}")
        processor = WhisperProcessor.from_pretrained(base_model_name, language=LANGUAGE, task=TASK)

    # Base model + MAS 주입
    print(f"[LOAD] base model={base_model_name}")
    model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    model.config.pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    model.config.use_cache = False

    inject_mas_encoder(model, targets=TARGETS, r=MAS_R, alpha=MAS_ALPHA,
                       dropout=MAS_DROPOUT, n_experts=N_EXPERTS)

    # state_dict 로드
    print(f"[LOAD] MAS state_dict from {state_path}")
    state = torch.load(state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] missing keys: {missing[:10]} ... ({len(missing)} keys)")
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected[:10]} ... ({len(unexpected)} keys)")

    model.to(device).eval()
    return processor, model


# ── MAS 디코딩 (도메인별 gate 설정 포함) ─────────────────────────────────────

@torch.no_grad()
def batch_generate_mas(model,
                       input_features: torch.Tensor,
                       processor: WhisperProcessor,
                       device: str,
                       metas: List[Dict[str, Any]],
                       max_new_tokens: int,
                       language: Optional[str],
                       task: str) -> List[str]:
    """
    metas 안의 domain을 보고 gate 설정:
    - Tel 은 애초에 필터링되어 들어오지 않는다고 가정.
    - <Dia><*> 또는 <Day> → 5 expert 균등 가중치
    - 그 외(<News> 등)       → gate=0 → base(단일 LoRA merge)만 사용
    """
    B = len(metas)
    domains = [m.get("domain") for m in metas]

    W = torch.zeros(B, N_EXPERTS, device=device)
    for i, dom in enumerate(domains):
        if dom in ACTIVE_DIALECTS or dom == "<Day>":
            W[i, :] = 1.0 / N_EXPERTS  # 균등 mixture
        else:
            W[i, :] = 0.0              # MAS residual off → base only

    set_all_mas_weights(model, W)

    kwargs = {"max_new_tokens": max_new_tokens}
    if language:
        kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
            language=language, task=task
        )
    gen = model.generate(inputs=input_features.to(device), **kwargs)
    texts = processor.batch_decode(gen, skip_special_tokens=True)
    return [t.strip() for t in texts]


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    # JSON 로드
    items = load_items(JSON_PATH)
    if not items:
        print("[WARN] 입력 JSON 비어있음")
        return

    # Tel 필터링
    if EXCLUDE_TEL:
        n_before = len(items)
        tel_before = sum(1 for r in items if is_tel_item(r))
        items = [r for r in items if not is_tel_item(r)]
        n_after = len(items)
        print(f"[Filter] items: {n_before} -> {n_after} | Tel: {tel_before} -> 0")
    else:
        print("[Filter] EXCLUDE_TEL=False, Tel 포함 평가")

    # Dataset / DataLoader
    ds = IdOnlyDataset(items, AUDIO_ROOT, PATH_TEMPLATE)
    collate = Collator(target_sr=TARGET_SR)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        collate_fn=collate,
    )

    # MAS 모델 로드
    processor, mas_model = load_mas_model(
        BASE_MODEL_NAME, MAS_MODEL_DIR, MAS_STATE_PATH, DEVICE
    )

    outputs: List[Dict[str, Any]] = []
    missing_count = 0

    for batch in loader:
        if batch["dropped"]:
            missing_count += len(batch["dropped"])
        if not batch["waves"]:
            continue

        waves_list = [w.numpy() for w in batch["waves"]]
        inputs = processor(
            audio=waves_list,
            sampling_rate=TARGET_SR,
            return_tensors="pt"
        )
        feats = inputs["input_features"]  # (B, 80, T)

        preds = batch_generate_mas(
            mas_model,
            feats,
            processor,
            DEVICE,
            batch["metas"],
            MAX_NEW_TOKENS,
            LANGUAGE,
            TASK,
        )

        # ★ JSON 출력 형식 tel_off_lora_to_json.py와 동일하게 유지
        for meta, y in zip(batch["metas"], preds):
            outputs.append({
                "id": meta["id"],
                "domain": meta["domain"],
                "text_ref": meta["ref"],
                "text_pred_lora": y,
                "audio_path": meta["audio_path"],
            })

    # 저장
    outdir = os.path.join(OUTPUT_DIR, RUN_NAME)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "pred_small_mas_lora.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"[OK] 저장: {outpath}")
    if missing_count:
        print(f"[INFO] 누락 오디오 {missing_count}건 건너뜀 (AUDIO_ROOT/PATH_TEMPLATE 확인)")

if __name__ == "__main__":
    main()
