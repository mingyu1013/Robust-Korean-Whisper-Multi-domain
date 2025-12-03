# LoRA_train.py
# -*- coding: utf-8 -*-
"""
Whisper-small + Vanilla LoRA(q,v) — 공용 유틸(/home/work/cymg0001/preprocessed_audio/utils/kwhisper.py) 사용
- 통 LoRA(Q/V: encoder self, decoder self, decoder cross)
- EOS 보장 + PAD→-100 마스킹(공용유틸)으로 '오디오 끝 이후 계속 생성' 현상 완화
- 100 epochs + EarlyStopping(patience=8, delta=1e-6)
- LoRA-only 검증 로그 & 커버리지 감사
- 실행: python vanilla_lora_train.py
"""

import os, json, time, math, logging, random
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration, get_scheduler
from peft import get_peft_model, LoraConfig, TaskType

# ─────────────────────────────────────────────────────────
# 경로에 공용 유틸 추가 후 import
# ─────────────────────────────────────────────────────────
import sys
sys.path.append("/home/work/cymg0001/preprocessed_audio/utils")

import kwhisper as kw


# ─────────────────────────────────────────────────────────
# 기본 설정 (필요 시 아래만 수정)
# ─────────────────────────────────────────────────────────
DEFAULT_DATA_DIR   = "/home/work/cymg0001/preprocessed_audio/sml_arrow_fixed_train"   # Arrow load_from_disk 경로
DEFAULT_OUTPUT_DIR = "/home/work/cymg0001/preprocessed_audio/runs_lora"       # 산출물 폴더
DEFAULT_RUN_NAME   = "tel_off_lora_small"

DEFAULT_MODEL_NAME = "openai/whisper-small"
DEFAULT_LANGUAGE   = "ko"
DEFAULT_TASK       = "transcribe"

# 도메인 필터 설정(필요 시 확장)
EXCLUDE_DOMAIN_IDS = {1}          # {1}이면 Tel만 제외
EXCLUDE_DOMAIN_TOKENS = {"<Tel>"} # 문자열 라벨로 저장된 경우 대비

# 학습 하이퍼
EPOCHS             = 100
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

# LoRA(바닐라)
LORA_R             = 16
LORA_ALPHA         = 32.0
LORA_DROPOUT       = 0.1
LORA_TARGETS       = "q_proj,v_proj"   # 필요 시 "q_proj,k_proj,v_proj,out_proj"

# 저장 on/off
SAVE_MODE          = True              # True면 best/final/로그 저장

# 로깅
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vanilla-lora-train")

# ─────────────────────────────────────────────────────────
# 재현성/AMP
# ─────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def get_amp_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

# ─────────────────────────────────────────────────────────
# 로깅/저장 유틸
# ─────────────────────────────────────────────────────────
def save_adapter_and_proc(model, processor, outdir, tag, meta=None):
    path = os.path.join(outdir, tag)
    os.makedirs(path, exist_ok=True)
    try: model.save_pretrained(path)
    except Exception: pass
    try: processor.save_pretrained(os.path.join(path, "processor"))
    except Exception: pass
    if meta:
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"[SAVED] {tag} -> {path}")

# ─────────────────────────────────────────────────────────
# LoRA-only 점검/커버리지 감사
# ─────────────────────────────────────────────────────────
def print_lora_targets(model):
    hits = [n for n,_ in model.named_modules() if any(x in n for x in ["q_proj","k_proj","v_proj","out_proj"])]
    logger.info(f"[LoRA targets] matched modules: {len(hits)} (sample: {hits[:8]})")

def log_trainable_parameters(model, out_path=None, max_show=30, assert_only_lora=True):
    total, trainable = 0, 0
    train_names, non_lora = [], []
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            train_names.append(n)
            if "lora_" not in n:
                non_lora.append(n)
    ratio = (trainable / total * 100) if total else 0.0
    logger.info(f"[Params] total={total:,}  trainable={trainable:,} ({ratio:.4f}%)")
    if train_names:
        logger.info("[Trainable names sample]\n  - " + "\n  - ".join(train_names[:max_show]))
    if non_lora:
        logger.error("[ASSERT FAIL] Non-LoRA trainables detected (sample): " + ", ".join(non_lora[:10]))
        if assert_only_lora:
            raise AssertionError("LoRA-only 위반: non-LoRA 파라미터가 학습 가능 상태입니다.")
    else:
        logger.info("[OK] Only LoRA params are trainable.")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_params": total,
                "trainable_params": trainable,
                "trainable_ratio": trainable/total if total else 0.0,
                "sample_trainable": train_names[:max_show],
                "non_lora_trainables": non_lora[:max_show]
            }, f, indent=2, ensure_ascii=False)

def audit_lora_coverage(model, max_show=6):
    import re
    buckets = {"enc.self": [], "dec.self": [], "dec.cross": []}
    for name, module in model.named_modules():
        has_lora = hasattr(module, "lora_A") and hasattr(module, "lora_B")
        if not has_lora:
            continue
        if ".encoder.layers." in name and ".self_attn." in name: buckets["enc.self"].append(name)
        elif ".decoder.layers." in name and ".self_attn." in name: buckets["dec.self"].append(name)
        elif ".decoder.layers." in name and ".encoder_attn." in name: buckets["dec.cross"].append(name)

    print("\n[LoRA coverage by scope]")
    for k in ["enc.self", "dec.self", "dec.cross"]:
        v = buckets[k]
        print(f" - {k:9s}: {len(v):3d} modules", ("" if not v else f"(sample: {v[:max_show]})"))

    trainables = [n for n,p in model.named_parameters() if p.requires_grad]
    def has_trainable(scope_key):
        pattern = {
            "enc.self":  r"\.encoder\.layers\.\d+\.self_attn\.(q_proj|v_proj)\.lora_(A|B)\.",
            "dec.self":  r"\.decoder\.layers\.\d+\.self_attn\.(q_proj|v_proj)\.lora_(A|B)\.",
            "dec.cross": r"\.decoder\.layers\.\d+\.encoder_attn\.(q_proj|v_proj)\.lora_(A|B)\.",
        }[scope_key]
        rgx = re.compile(pattern)
        return any(rgx.search(n) for n in trainables)

    missing = [k for k in buckets if len(buckets[k]) == 0 or not has_trainable(k)]
    if missing:
        msgs = "; ".join(missing)
        raise AssertionError(f"[ASSERT FAIL] LoRA 주입/학습 누락 감지 — {msgs}")
    else:
        print("[OK] Encoder self / Decoder self / Decoder cross — LoRA(q,v) 주입 및 학습 가능 상태.\n")

# ─────────────────────────────────────────────────────────
# 메인 루프
# ─────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    kw.patch_whisper_forward_once()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = get_amp_dtype()

    outdir = os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_RUN_NAME)
    os.makedirs(outdir, exist_ok=True)
    log_dir = os.path.join(outdir, "logs"); os.makedirs(log_dir, exist_ok=True)
    epoch_log_path = os.path.join(log_dir, "epoch_log.json")
    trainable_report_path = os.path.join(log_dir, "trainable_report.json")

    # Processor/Model
    processor = WhisperProcessor.from_pretrained(DEFAULT_MODEL_NAME, language=DEFAULT_LANGUAGE, task=DEFAULT_TASK)
    model = WhisperForConditionalGeneration.from_pretrained(DEFAULT_MODEL_NAME)
    model.config.pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    model.config.use_cache = False
    model.to(device)

    # 데이터 로드 → EOS 보장 + labels 생성(없으면 text→tokenize)
    logger.info(f"[Data] load_from_disk: {DEFAULT_DATA_DIR}")
    ds_all = load_from_disk(DEFAULT_DATA_DIR)

    # ----- 도메인 필터: <Tel>(id=1) 제외 -----
    def _keep_example(ex):
        for k in ("domain_id", "domain_idx", "domain_label"):
            if k in ex and isinstance(ex[k], int):
                return ex[k] not in EXCLUDE_DOMAIN_IDS
        if "domain" in ex and isinstance(ex["domain"], str):
            return ex["domain"] not in EXCLUDE_DOMAIN_TOKENS
        return True

    kw.patch_whisper_forward_once()
    logger.info("[kwhisper] forward patched once")

    def _is_tel(ex):
        for k in ("domain_id", "domain_idx", "domain_label"):
            if k in ex and isinstance(ex[k], int):
                return ex[k] == 1
        if "domain" in ex and isinstance(ex["domain"], str):
            return ex["domain"] == "<Tel>"
        return False

    n_total_before = ds_all.num_rows
    n_tel_before = ds_all.filter(_is_tel).num_rows

    ds_all = ds_all.filter(_keep_example)

    n_total_after = ds_all.num_rows
    n_tel_after = ds_all.filter(_is_tel).num_rows

    logger.info(f"[Filter] total: {n_total_before} -> {n_total_after} | Tel: {n_tel_before} -> {n_tel_after}")

    # ----- 라벨/EOS 보장 및 분할 -----
    ds_all = kw.ensure_labels_and_eos(ds_all, processor.tokenizer, text_key="text", labels_key="labels")
    split = ds_all.train_test_split(test_size=VAL_RATIO, seed=SEED)
    train_ds, val_ds = split["train"], split["test"]
    logger.info(f"[Data] Train={len(train_ds)}  Val={len(val_ds)} (ratio={VAL_RATIO})")

    cols = ["input_features", "attention_mask", "labels"]
    train_ds = train_ds.with_format("torch", columns=cols)
    val_ds   = val_ds.with_format("torch", columns=cols)

    # Collator (공용유틸: pad로만 패딩)
    collate = kw.make_data_collator(processor)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate)

    # LoRA(q,v) 주입
    target_modules = [x.strip() for x in LORA_TARGETS.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        target_modules=target_modules, task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_cfg).to(device)

    # LoRA-only 체크 + 커버리지 감사
    print_lora_targets(model)
    log_trainable_parameters(model, out_path=trainable_report_path, assert_only_lora=True)
    audit_lora_coverage(model)

    # Optim/Sched
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train_loader) / max(1, GRAD_ACCUM_STEPS))
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    PAD_ID = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    # 로그 상태
    epoch_logs: List[Dict] = []
    best_val = float("inf"); best_epoch = -1; no_improve = 0
    global_step = 0

    logger.info("▶ 학습 시작 — Vanilla LoRA(q,v), epochs=%d, ES(patience=%d, delta=%.1e)", EPOCHS, ES_PATIENCE, ES_DELTA)

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        model.train()
        tr_loss_sum = 0.0
        optimizer.zero_grad(set_to_none=True)

        # ===== Train =====
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = kw.labels_pad_to_ignore_by_len(batch["labels"], batch["labels_len"])
            with torch.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=amp_dtype):
                out = model(input_features=batch["input_features"], attention_mask=batch["attention_mask"], labels=labels)
                loss = out.loss / GRAD_ACCUM_STEPS
            loss.backward()
            tr_loss_sum += loss.item()

            if step % GRAD_ACCUM_STEPS == 0:
                if MAX_GRAD_NORM:
                    nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
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
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                labels = kw.labels_pad_to_ignore_by_len(batch["labels"], batch["labels_len"])
                with torch.autocast(device_type="cuda" if device=="cuda" else "cpu", dtype=amp_dtype):
                    out = model(input_features=batch["input_features"], attention_mask=batch["attention_mask"], labels=labels)
                    val_loss_sum += out.loss.item()
        val_loss_epoch = val_loss_sum / max(1, len(val_loader))

        dt = time.time() - t0
        improved = (best_val - val_loss_epoch) > ES_DELTA
        if improved:
            best_val, best_epoch = val_loss_epoch, epoch
            no_improve = 0
            status = "↑ best update"
            if SAVE_MODE:
                save_adapter_and_proc(model, processor, outdir, "best",
                                      meta={"epoch": epoch, "val_loss": float(val_loss_epoch), "train_loss": float(train_loss_epoch)})
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
            logger.info("[EARLY STOP] patience=%d 도달 — best@%d (val=%.4f)", ES_PATIENCE, best_epoch, best_val)
            break

        if device == "cuda":
            torch.cuda.empty_cache()

    # final 저장
    if SAVE_MODE:
        tag_meta = {"epoch": epoch, "val_loss": float(val_loss_epoch), "best_epoch": best_epoch, "best_val": float(best_val)}
        save_adapter_and_proc(model, processor, outdir, "final", meta=tag_meta)

    logger.info("✅ 완료. Best@%d val=%.4f", best_epoch, best_val)
    logger.info("🗂 Logs: %s", epoch_log_path)
    logger.info("🗂 Trainable report: %s", trainable_report_path)

if __name__ == "__main__":
    main()
