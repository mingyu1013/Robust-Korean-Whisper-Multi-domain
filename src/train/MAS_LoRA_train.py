# -*- coding: utf-8 -*-
"""
dialect_mas_lora.py

Stage2:
  - Stage1 단일 LoRA 어댑터를 Whisper-small에 merge → 공통 베이스(단일 LoRA까지 포함된 모델)로 사용
  - 이 베이스는 전부 freeze
  - Encoder self-attn q_proj, v_proj 에만 MAS-LoRA(방언 전용 expert 5개) 주입

데이터:
  - <Day>, <News>, <Tel> 등은 전부 제거
  - <Dia><JL>, <Dia><GW>, <Dia><GS>, <Dia><JJ>, <Dia><CC> 5개 방언만 사용

학습:
  - 각 방언 샘플은 자기 expert만 1인 one-hot gate (accent-aware)
  - Optimizer는 MAS-LoRA의 A/B(As., Bs.) 파라미터만 업데이트
  - 검증은 방언에 대해 5 expert 균등(1/5) mixture로 평가 (나중에 inference 모드랑 맞추기 좋게)
"""

import os, json, time, math, logging, random
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration, get_scheduler
from peft import PeftModel

import sys
sys.path.append("/home/work/cymg0001/preprocessed_audio/utils")
import kwhisper as kw

# ─────────────────────────────────────────────────────────
# 경로 / 기본 설정
# ─────────────────────────────────────────────────────────

DATA_DIR        = "/home/work/cymg0001/preprocessed_audio/sml_arrow_fixed_train"
OUTPUT_DIR      = "/home/work/cymg0001/preprocessed_audio/mg/dialect_mas_lora_output"
RUN_NAME        = "mas_lora_small_dialect_from_single"

BASE_MODEL_NAME = "openai/whisper-small"
LANGUAGE        = "ko"
TASK            = "transcribe"

# Stage1 단일 LoRA 어댑터 경로
SINGLE_LORA_DIR = "/home/work/cymg0001/preprocessed_audio/runs_lora/tel_off_lora_small/best"

# Processor: Stage1에서 저장한 processor가 있으면 그거 쓰고, 없으면 base에서 로딩
PROCESSOR_DIR   = os.path.join(SINGLE_LORA_DIR, "processor")

# 학습 설정
EPOCHS             = 60
BATCH_SIZE         = 8
VAL_BATCH_SIZE     = 8
GRAD_ACCUM_STEPS   = 4
LEARNING_RATE      = 3e-5
WEIGHT_DECAY       = 0.0
WARMUP_FRAC        = 0.10
MAX_GRAD_NORM      = 1.0
VAL_RATIO          = 0.05
SEED               = 42

# Early Stopping
ES_PATIENCE        = 8
ES_DELTA           = 1e-6

# 방언 도메인 / expert 매핑
ACTIVE_DIALECTS = [
    "<Dia><JL>",
    "<Dia><GW>",
    "<Dia><GS>",
    "<Dia><JJ>",
    "<Dia><CC>",
]
DOMAIN2IDX: Dict[str, int] = {d: i for i, d in enumerate(ACTIVE_DIALECTS)}
N_EXPERTS = len(ACTIVE_DIALECTS)

# MAS-LoRA 하이퍼 (encoder only)
MAS_R       = 8
MAS_ALPHA   = 16.0
MAS_DROPOUT = 0.1
TARGETS     = ("q_proj", "v_proj")

# 로깅
SAVE_MODE        = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mas-lora-dialect")

# ─────────────────────────────────────────────────────────
# 재현성 / AMP
# ─────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def get_amp_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

# ─────────────────────────────────────────────────────────
# MAS-LoRA 모듈 (Stage1 단일 LoRA merge된 Linear 위에 expert residual)
# ─────────────────────────────────────────────────────────

class MASLoRALinear(nn.Module):
    """
    - base: Stage1 단일 LoRA까지 merge된 nn.Linear (freeze)
    - As/Bs: 각 expert 별 LoRA 파라미터 (trainable)
    - _expert_weights: (B, n_experts) — 외부에서 설정하는 gate (one-hot / uniform 등)
    """
    def __init__(self, base: nn.Linear, r: int, alpha: float,
                 dropout: float, n_experts: int):
        super().__init__()
        assert isinstance(base, nn.Linear), "MASLoRALinear base는 nn.Linear 여야 합니다."
        self.base = base
        self.n_experts = int(n_experts)
        self.r = int(r)
        self.scaling = alpha / max(1, self.r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f  = base.in_features
        out_f = base.out_features
        self.in_features  = in_f
        self.out_features = out_f

        self.As = nn.ParameterList([
            nn.Parameter(torch.zeros(self.r, in_f))
            for _ in range(self.n_experts)
        ])
        self.Bs = nn.ParameterList([
            nn.Parameter(torch.zeros(out_f, self.r))
            for _ in range(self.n_experts)
        ])
        self.reset_parameters()

        self._expert_weights: Optional[torch.Tensor] = None

    def reset_parameters(self):
        for A, B in zip(self.As, self.Bs):
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            nn.init.zeros_(B)

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


def inject_mas_encoder(model: nn.Module,
                       targets=("q_proj", "v_proj"),
                       r=8, alpha=16.0, dropout=0.1, n_experts=5):
    """
    Encoder self-attn q_proj/v_proj 자리에 MASLoRALinear 주입.
    base는 현재 Linear (Stage1 로라 merge된 상태)를 그대로 사용.
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
        if not isinstance(sub, nn.Linear):
            continue

        mas = MASLoRALinear(sub, r=r, alpha=alpha, dropout=dropout, n_experts=n_experts)
        setattr(parent, attr, mas)
        replaced += 1

    logger.info(f"[MAS-ENC] injected {replaced} modules on encoder.self targets={targets}")


def set_all_mas_weights(model: nn.Module, w: Optional[torch.Tensor]):
    """
    모델 안의 모든 MASLoRALinear 모듈에 동일한 gate w(B, E)를 설정
    """
    for m in model.modules():
        if isinstance(m, MASLoRALinear):
            m.set_expert_weights(w)

# ─────────────────────────────────────────────────────────
# 저장 유틸
# ─────────────────────────────────────────────────────────

def save_model_and_proc(model, processor, outdir, tag, meta=None):
    path = os.path.join(outdir, tag)
    os.makedirs(path, exist_ok=True)
    try:
        model.save_pretrained(path)
    except Exception:
        pass
    try:
        processor.save_pretrained(os.path.join(path, "processor"))
    except Exception:
        pass
    if meta:
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"[SAVED] {tag} -> {path}")

    # MAS 파라미터 포함 state_dict도 추가로 저장
    try:
        torch.save(model.state_dict(), os.path.join(path, "pytorch_mas.bin"))
    except Exception:
        logger.warning("state_dict 저장 실패 (무시해도 됨).")

# ─────────────────────────────────────────────────────────
# 도메인 필터 / mas_idx 부여
# ─────────────────────────────────────────────────────────

def _keep_dialect_only(ex: Dict[str, Any]) -> bool:
    """
    방언 도메인 5개만 남김.
    <Day>, <News>, <Tel> 등 방언이 아닌 도메인은 모두 제거.
    """
    dom = ex.get("domain", None)
    return isinstance(dom, str) and dom in ACTIVE_DIALECTS

def _assign_mas_idx(ex: Dict[str, Any]) -> Dict[str, Any]:
    dom = ex.get("domain", None)
    ex["mas_idx"] = int(DOMAIN2IDX.get(dom, -1))
    return ex

# ─────────────────────────────────────────────────────────
# 메인 학습 루프
# ─────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    kw.patch_whisper_forward_once()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = get_amp_dtype()

    outdir = os.path.join(OUTPUT_DIR, RUN_NAME)
    os.makedirs(outdir, exist_ok=True)
    log_dir = os.path.join(outdir, "logs"); os.makedirs(log_dir, exist_ok=True)
    epoch_log_path = os.path.join(log_dir, "epoch_log.json")

    # Processor 로딩 (Stage1에서 저장한 processor 우선)
    if os.path.isdir(PROCESSOR_DIR):
        logger.info(f"[LOAD] processor from {PROCESSOR_DIR}")
        processor = WhisperProcessor.from_pretrained(PROCESSOR_DIR)
    else:
        logger.info(f"[LOAD] processor from base model {BASE_MODEL_NAME}")
        processor = WhisperProcessor.from_pretrained(BASE_MODEL_NAME, language=LANGUAGE, task=TASK)

    # Stage1 단일 LoRA merge → 공통 베이스
    logger.info(f"[LOAD] base model={BASE_MODEL_NAME}")
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
    base_model.config.pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    base_model.config.use_cache = False

    logger.info(f"[LOAD] Stage1 single LoRA from {SINGLE_LORA_DIR}")
    peft_model = PeftModel.from_pretrained(base_model, SINGLE_LORA_DIR)
    model = peft_model.merge_and_unload()
    model.to(device)

    # Encoder에 MAS-LoRA expert 5개 주입
    inject_mas_encoder(model, TARGETS, r=MAS_R, alpha=MAS_ALPHA,
                       dropout=MAS_DROPOUT, n_experts=N_EXPERTS)

    # 모든 파라미터 freeze 후, MAS As/Bs만 학습
    for n, p in model.named_parameters():
        p.requires_grad = False
    mas_params = []
    for n, p in model.named_parameters():
        if "As." in n or "Bs." in n:
            p.requires_grad = True
            mas_params.append(p)
    logger.info(f"[Params] MAS trainable params = {sum(p.numel() for p in mas_params):,}")

    # 데이터 로드
    logger.info(f"[Data] load_from_disk: {DATA_DIR}")
    ds_all = load_from_disk(DATA_DIR)

    # 방언 도메인만 남기기
    n_total_before = ds_all.num_rows
    ds_all = ds_all.filter(_keep_dialect_only)
    n_total_after = ds_all.num_rows
    logger.info(f"[Filter] total: {n_total_before} -> {n_total_after} (dialect only)")
    logger.info(f"[Filter] ACTIVE_DIALECTS={ACTIVE_DIALECTS}")

    # mas_idx 부여
    ds_all = ds_all.map(_assign_mas_idx)
    assert all(x != -1 for x in ds_all["mas_idx"]), "[ERROR] mas_idx=-1 (방언 매핑 실패) 존재"

    # 라벨/EOS 보장
    ds_all = kw.ensure_labels_and_eos(ds_all, processor.tokenizer,
                                      text_key="text", labels_key="labels")

    # Train/Val split
    split = ds_all.train_test_split(test_size=VAL_RATIO, seed=SEED)
    train_ds, val_ds = split["train"], split["test"]
    logger.info(f"[Data] Train={len(train_ds)}  Val={len(val_ds)} (ratio={VAL_RATIO})")

    # labels_len 생성 (원래 MAS 코드랑 동일 패턴)
    def _add_labels_len(batch):
        labs = batch["labels"]
        if hasattr(labs[0], "shape"):
            lens = [int(x.shape[0]) for x in labs]
        else:
            lens = [len(x) for x in labs]
        return {"labels_len": lens}

    train_ds = train_ds.map(_add_labels_len, batched=True)
    val_ds   = val_ds.map(_add_labels_len,   batched=True)

    # 도메인 분포 확인 (Train 기준)
    train_mas_idx = np.array(train_ds["mas_idx"], dtype=np.int64)
    counts = np.bincount(train_mas_idx, minlength=N_EXPERTS)
    p_domain = counts / counts.sum()
    logger.info(f"[Domain] counts={counts.tolist()}  p_domain={p_domain.tolist()} (order={ACTIVE_DIALECTS})")

    # 포맷 설정
    cols = ["input_features", "attention_mask", "labels", "labels_len", "mas_idx"]
    train_ds = train_ds.with_format("torch", columns=cols, output_all_columns=True)
    val_ds   = val_ds.with_format("torch",   columns=cols, output_all_columns=True)

    # Collator (기존 collator + mas_idx 붙이는 래퍼)
    base_collate = kw.make_data_collator(processor)

    def collate_with_mas(examples):
        # 기본 collate로 input_features, attention_mask, labels, labels_len 등 처리
        batch = base_collate(examples)
        # examples에서 mas_idx 모아서 텐서로 추가
        mas_idx_list = []
        for ex in examples:
            mas_idx_list.append(int(ex["mas_idx"]))
        batch["mas_idx"] = torch.tensor(mas_idx_list, dtype=torch.long)
        return batch

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_with_mas,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_with_mas,
    )

    # Optim / Scheduler (MAS 파라미터만)
    optimizer = torch.optim.AdamW(mas_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train_loader) / max(1, GRAD_ACCUM_STEPS))
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    PAD_ID = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    epoch_logs: List[Dict[str, Any]] = []
    best_val = float("inf"); best_epoch = -1; no_improve = 0
    global_step = 0

    logger.info(f"▶ Train start(MAS dialect) — epochs={EPOCHS}, experts={N_EXPERTS}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        tr_loss_sum = 0.0
        optimizer.zero_grad(set_to_none=True)

        # ===== Train =====
        for step, batch in enumerate(train_loader, start=1):
            mas_idx = batch.get("mas_idx", None)
            labels_len = batch.get("labels_len", None)

            if mas_idx is None:
                raise KeyError("batch['mas_idx']가 collator에서 전달되지 않았습니다. "
                               "collate_with_mas 구현을 확인하세요.")
            if labels_len is None:
                raise KeyError("batch['labels_len']가 없습니다. _add_labels_len / with_format 구현 확인 필요.")

            # Tensor만 device로
            batch_t = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                       for k, v in batch.items()}

            mas_idx_t = batch_t["mas_idx"]
            labels_len_t = batch_t["labels_len"]

            # one-hot gate (각 방언은 자기 expert만 ON)
            Bsz = mas_idx_t.shape[0]
            W = torch.zeros(Bsz, N_EXPERTS, device=device)
            W[torch.arange(Bsz, device=device), mas_idx_t] = 1.0
            set_all_mas_weights(model, W)

            # labels pad→-100
            labels = kw.labels_pad_to_ignore_by_len(batch_t["labels"], labels_len_t)

            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=amp_dtype):
                out = model(
                    input_features=batch_t["input_features"],
                    attention_mask=batch_t.get("attention_mask", None),
                    labels=labels,
                )
                loss = out.loss / GRAD_ACCUM_STEPS

            loss.backward()
            tr_loss_sum += loss.item()

            if step % GRAD_ACCUM_STEPS == 0:
                if MAX_GRAD_NORM:
                    nn.utils.clip_grad_norm_(mas_params, MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        train_loss_epoch = (tr_loss_sum / max(1, steps_per_epoch)) * GRAD_ACCUM_STEPS

        # ===== Validation =====
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                labels_len = batch.get("labels_len", None)
                if labels_len is None:
                    raise KeyError("val batch 에 labels_len 누락 — train과 동일하게 _add_labels_len/with_format 필요.")

                batch_t = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                           for k, v in batch.items()}
                labels_len_t = batch_t["labels_len"]

                # ★ 검증 시에는 방언에 대해 5 expert 균등(1/5) mixture로 평가
                Bsz = batch_t["labels"].shape[0]
                W = torch.full((Bsz, N_EXPERTS), 1.0 / N_EXPERTS, device=device)
                set_all_mas_weights(model, W)

                labels = kw.labels_pad_to_ignore_by_len(batch_t["labels"], labels_len_t)

                with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=amp_dtype):
                    out = model(
                        input_features=batch_t["input_features"],
                        attention_mask=batch_t.get("attention_mask", None),
                        labels=labels,
                    )
                    val_loss_sum += out.loss.item()

        val_loss_epoch = val_loss_sum / max(1, len(val_loader))
        dt = time.time() - t0

        improved = (best_val - val_loss_epoch) > ES_DELTA
        if improved:
            best_val, best_epoch = val_loss_epoch, epoch
            no_improve = 0
            status = "↑ best"
            if SAVE_MODE:
                save_model_and_proc(
                    model, processor, outdir, "best",
                    meta={
                        "epoch": epoch,
                        "val_loss": float(val_loss_epoch),
                        "train_loss": float(train_loss_epoch),
                        "p_domain": p_domain.tolist(),
                        "domains": ACTIVE_DIALECTS,
                    }
                )
        else:
            no_improve += 1
            status = f"→ no improve ({no_improve}/{ES_PATIENCE})"

        logger.info("[Epoch %03d] train_loss=%.4f | val_loss=%.4f | %s | %.1fs",
                    epoch, train_loss_epoch, val_loss_epoch, status, dt)

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": float(train_loss_epoch),
            "val_loss": float(val_loss_epoch),
            "global_step": int(global_step),
            "no_improve": int(no_improve),
            "time_sec": float(dt),
        })
        with open(epoch_log_path, "w", encoding="utf-8") as f:
            json.dump(epoch_logs, f, ensure_ascii=False, indent=2)

        if ES_PATIENCE and no_improve >= ES_PATIENCE:
            logger.info("[EARLY STOP] best@%d (val=%.4f)", best_epoch, best_val)
            break

        if device == "cuda":
            torch.cuda.empty_cache()

    # final 저장
    if SAVE_MODE:
        meta = {
            "epoch": epoch,
            "val_loss": float(val_loss_epoch),
            "best_epoch": best_epoch,
            "best_val": float(best_val),
        }
        save_model_and_proc(model, processor, outdir, "final", meta)

    logger.info("✅ 완료. Best@%d val=%.4f", best_epoch, best_val)
    logger.info("🗂 Logs: %s", epoch_log_path)


if __name__ == "__main__":
    main()
