# kwhisper.py
# -*- coding: utf-8 -*-
"""
Whisper 학습 공용 유틸 (길이 기반 라벨 마스킹으로 EOS 보존)
- labels 생성(없으면 text→tokenize) + EOS 보장
- Collator: pad_token으로 패딩 + 각 샘플 라벨 길이(labels_len) 함께 반환
- Loss 직전: 길이 기반으로 진짜 패딩만 -100 처리( pad_id==eos_id 문제 방지 )
- Whisper forward 패치(중복 방지)
"""

import logging
from typing import List, Dict, Any
from datasets import Dataset
import torch as T

_logger = logging.getLogger("kwhisper")

# ─────────────────────────────────────────────
# Whisper forward 패치 (한 번만)
# ─────────────────────────────────────────────
_PATCHED = False
def patch_whisper_forward_once():
    global _PATCHED
    if _PATCHED: return
    try:
        from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration as _WhisperClass
        _orig_forward = _WhisperClass.forward
        def _patched_forward(self, *args, **kwargs):
            for k in ["input_ids","inputs_embeds","decoder_input_ids","decoder_inputs_embeds",
                      "num_items_in_batch","return_outputs"]:
                kwargs.pop(k, None)
            return _orig_forward(self, *args, **kwargs)
        _WhisperClass.forward = _patched_forward
        _PATCHED = True
        _logger.info("[kwhisper] Patched Whisper.forward")
    except Exception as e:
        _logger.warning(f"[kwhisper] forward patch skipped: {e}")

# ─────────────────────────────────────────────
# labels 생성(없으면 text→tokenize) + EOS 보장
# ─────────────────────────────────────────────
def ensure_labels_and_eos(ds: Dataset, tokenizer, text_key="text", labels_key="labels") -> Dataset:
    eos = tokenizer.eos_token_id

    # labels가 없으면 text로부터 생성(+EOS)
    if labels_key not in ds.column_names and text_key in ds.column_names:
        def _mk(ex):
            ids = tokenizer(ex[text_key], add_special_tokens=False)["input_ids"]
            if eos is not None and (len(ids) == 0 or ids[-1] != eos):
                ids = ids + [eos]
            ex[labels_key] = ids
            return ex
        ds = ds.map(_mk, desc="[kwhisper] build labels from text")

    # labels가 있어도 EOS 보장
    def _eos(ex):
        ids = ex[labels_key]
        if len(ids) == 0 or (eos is not None and ids[-1] != eos):
            ids = ids + [eos]
        ex[labels_key] = ids
        return ex
    ds = ds.map(_eos, desc="[kwhisper] ensure EOS on labels")
    return ds

# ─────────────────────────────────────────────
# Collator: pad_token으로 패딩 + 각 샘플 라벨 길이 반환
#   ※ pad_id==eos_id라도 길이로 패딩을 구분할 수 있게 함
# ─────────────────────────────────────────────
def make_data_collator(processor):
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    def collator(batch: List[Dict[str, Any]]):
        import torch as T
        def to_tensor(x, dtype):
            if isinstance(x, T.Tensor): return x.to(dtype)
            return T.as_tensor(x, dtype=dtype)

        feats = T.stack([to_tensor(b["input_features"], T.float32) for b in batch])
        masks = T.stack([to_tensor(b["attention_mask"], T.long) for b in batch])

        # 라벨과 원래 길이 저장
        labs_list, lens = [], []
        for b in batch:
            l = b["labels"]
            if isinstance(l, T.Tensor): l = l.tolist()
            lens.append(len(l))
            labs_list.append(T.as_tensor(l, dtype=T.long))

        labels = T.nn.utils.rnn.pad_sequence(labs_list, batch_first=True, padding_value=pad_id)
        labels_len = T.as_tensor(lens, dtype=T.long)   # [B]

        # domain_label이 있으면 함께 반환
        out = {"input_features": feats, "attention_mask": masks, "labels": labels, "labels_len": labels_len}
        if "domain_label" in batch[0]:
            out["domain_labels"] = T.as_tensor([int(b["domain_label"]) for b in batch], dtype=T.long)
        return out
    return collator

# ─────────────────────────────────────────────
# 길이 기반 라벨 마스킹: 진짜 패딩만 -100
#   labels: [B, T] (pad로 패딩됨), labels_len: [B]
# ─────────────────────────────────────────────
def labels_pad_to_ignore_by_len(labels: T.Tensor, labels_len: T.Tensor) -> T.Tensor:
    """
    각 샘플 i에 대해 positions >= labels_len[i] 만 -100 처리.
    pad_id==eos_id라도 내부 EOS는 그대로 학습됨.
    """
    B, Tm = labels.shape
    device = labels.device
    arange = T.arange(Tm, device=device).unsqueeze(0).expand(B, Tm)  # [B, Tm]
    valid = arange < labels_len.unsqueeze(1)                         # [B, Tm] bool
    masked = labels.clone()
    masked[~valid] = -100
    return masked
