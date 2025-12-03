# -*- coding: utf-8 -*-
# Whisper+LoRA+SML용 Arrow 전처리(스트리밍 저장: writer_batch_size로 안전하게 청크 분할)
import os, json, argparse, warnings
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import torchaudio
import torch
from tqdm import tqdm
import datasets
from datasets import Dataset, Features, Value, Sequence, Array2D
from transformers import WhisperFeatureExtractor  # 가볍게 FE만 사용

# ==== 고정 경로/설정(필요시 여기만 수정) =====================================
JSON_PATH  = "/home/work/cymg0001/preprocessed_audio/train/merged_train2.json"
AUDIO_DIR  = "/home/work/cymg0001/preprocessed_audio/train"
ARROW_OUT  = "/home/work/cymg0001/preprocessed_audio/sml_arrow_fixed_train"
FE_NAME    = "openai/whisper-tiny"  # FE 파라미터(멜 설정)만 사용. Tiny/Small 동일 전처리
LANGUAGE   = "ko"
TASK       = "transcribe"
SAMPLE_RATE = 16000
TARGET_MEL_FRAMES = 3000
MIN_TEXT_LEN = 3

# 저장 품질/용량 트레이드오프: float16로 저장하면 용량 1/2 (학습 시 collator에서 float32로 캐스트)
STORE_FP16 = False
WRITER_BATCH_SIZE = 256  # 청크 크기(한 번에 이만큼만 Arrow에 씀 → offset overflow 방지)
# ===========================================================================

def add_common_save_flag(parser, default_save='on'):
    parser.add_argument('--save', choices=['on','off'], default=default_save,
                        help="on: 산출물 저장, off: 저장 안 함")
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--run_name', type=str, default='sml_arrow_rebuild_stream')
    return parser

def should_save(args) -> bool:
    return getattr(args, 'save', 'on') == 'on'

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")

def find_audio(audio_dir: str, audio_id: str) -> Optional[str]:
    p = os.path.join(audio_dir, audio_id + ".wav")
    if os.path.isfile(p): return p
    for ext in AUDIO_EXTS:
        p = os.path.join(audio_dir, audio_id + ext)
        if os.path.isfile(p): return p
    for root, _, files in os.walk(audio_dir):
        for f in files:
            name, ext = os.path.splitext(f)
            if name == audio_id and ext.lower() in AUDIO_EXTS:
                return os.path.join(root, f)
    return None

def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "JSON은 리스트여야 합니다."
    for i, x in enumerate(data[:3]):
        for k in ["id","text","domain"]:
            if k not in x:
                raise ValueError(f"JSON 항목에 '{k}' 없음 (예: idx {i})")
    return data

@dataclass
class Cfg:
    sample_rate: int = SAMPLE_RATE
    target_mel_frames: int = TARGET_MEL_FRAMES
    min_text_len: int = MIN_TEXT_LEN
    language: str = LANGUAGE
    task: str = TASK

def load_audio_mono(path: str, sr: int) -> torch.Tensor:
    wav, in_sr = torchaudio.load(path)  # [ch, N]
    if wav.dtype != torch.float32:
        wav = wav.float()
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if in_sr != sr:
        wav = torchaudio.functional.resample(wav, in_sr, sr)
    return wav.squeeze(0)  # [N]

def extract_features(fe, audio: torch.Tensor, cfg: Cfg):
    out = fe(
        audio.numpy(),
        sampling_rate=cfg.sample_rate,
        return_tensors="np",
        return_attention_mask=True
    )
    feats = out.input_features[0]   # [80, T]
    amask = out.attention_mask[0]   # [T]

    T = feats.shape[-1]
    tgt = cfg.target_mel_frames
    if T < tgt:
        pad_f = np.zeros((feats.shape[0], tgt - T), dtype=feats.dtype)
        feats = np.concatenate([feats, pad_f], axis=1)
        pad_m = np.zeros((tgt - amask.shape[0],), dtype=amask.dtype)
        amask = np.concatenate([amask, pad_m], axis=0)
    elif T > tgt:
        feats = feats[:, :tgt]
        amask = amask[:tgt]

    amask = (amask > 0).astype("int64")
    if STORE_FP16:
        feats = feats.astype("float16")
    else:
        feats = feats.astype("float32")
    return feats, amask

def build_domain_map(items: List[Dict[str, Any]]) -> Dict[str, int]:
    uniq, seen = [], set()
    for x in items:
        d = x.get("domain", "<UNK>")
        if d not in seen:
            seen.add(d); uniq.append(d)
    return {d:i for i,d in enumerate(uniq)}

def main():
    parser = argparse.ArgumentParser()
    add_common_save_flag(parser, default_save='on')
    parser.add_argument('--json_path', type=str, default=JSON_PATH)
    parser.add_argument('--audio_dir', type=str, default=AUDIO_DIR)
    parser.add_argument('--arrow_out', type=str, default=ARROW_OUT)
    parser.add_argument('--fe_name', type=str, default=FE_NAME)
    parser.add_argument('--target_mel_frames', type=int, default=TARGET_MEL_FRAMES)
    parser.add_argument('--sample_rate', type=int, default=SAMPLE_RATE)
    parser.add_argument('--min_text_len', type=int, default=MIN_TEXT_LEN)
    args = parser.parse_args()

    cfg = Cfg(
        sample_rate=args.sample_rate,
        target_mel_frames=args.target_mel_frames,
        min_text_len=args.min_text_len
    )

    os.makedirs(args.arrow_out, exist_ok=True)
    err_log_path = os.path.join(args.arrow_out, "new_error_log.json")
    dom_map_path = os.path.join(args.arrow_out, "domain_map.json")
    err_log = []

    fe = WhisperFeatureExtractor.from_pretrained(args.fe_name)

    items = load_json(args.json_path)
    dom2id = build_domain_map(items)
    with open(dom_map_path, "w", encoding="utf-8") as f:
        json.dump(dom2id, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 총 샘플 ≈ {len(items)}. 멜 스펙트로그램 추출 및 스트리밍 저장 시작.")
    kept, missed, shorttxt, feat_err = 0, 0, 0, 0

    # 제너레이터: 하나씩 처리해 즉시 Arrow에 기록(큰 청크 금지)
    def gen():
        nonlocal kept, missed, shorttxt, feat_err
        for x in tqdm(items, desc="preprocess"):
            sid  = x["id"]
            text = (x.get("text") or "").strip()
            dom  = x.get("domain", "<UNK>")

            if len(text) < cfg.min_text_len:
                shorttxt += 1
                err_log.append({"id": sid, "reason": "short_text", "len": len(text)})
                continue

            wav_path = find_audio(args.audio_dir, sid)
            if wav_path is None:
                missed += 1
                err_log.append({"id": sid, "reason": "audio_not_found"})
                continue

            try:
                audio = load_audio_mono(wav_path, cfg.sample_rate)
                feats, amask = extract_features(fe, audio, cfg)
            except Exception as e:
                feat_err += 1
                err_log.append({"id": sid, "reason": "feature_extraction_error", "msg": repr(e)})
                continue

            kept += 1
            yield {
                "id": sid,
                "input_features": feats,
                "attention_mask": amask,
                "text": text,
                "domain": dom,
                "domain_label": int(dom2id.get(dom, dom2id.setdefault("<UNK>", len(dom2id))))
            }

    # 고정 스키마(우리 규격). input_features는 80×T 고정
    feats = Features({
        "id": Value("string"),
        "input_features": Array2D(dtype=("float16" if STORE_FP16 else "float32"),
                                  shape=(80, cfg.target_mel_frames)),
        "attention_mask": Sequence(Value("int64")),
        "text": Value("string"),
        "domain": Value("string"),
        "domain_label": Value("int64"),
    })

    # 스트리밍 생성 + 작은 writer_batch_size로 안전 저장
    ds = datasets.Dataset.from_generator(
        gen,
        features=feats,
        writer_batch_size=WRITER_BATCH_SIZE
    )

    print(f"[SUMMARY] kept={kept}, audio_missing={missed}, short_text={shorttxt}, feat_error={feat_err}, errors_logged={len(err_log)}")

    if should_save(args):
        ds.save_to_disk(args.arrow_out)
        with open(err_log_path, "w", encoding="utf-8") as f:
            json.dump(err_log, f, ensure_ascii=False, indent=2)
        print(f"[SAVED] dataset -> {args.arrow_out}")
        print(f"[SAVED] error log -> {err_log_path}")

    # 샘플 프린트
    for i in range(min(3, len(ds))):
        ex = ds[i]
        print(f"\n--- sample {i} ---")
        print("id:", ex["id"])
        arr = np.asarray(ex["input_features"])
        print("input_features shape:", arr.shape, arr.dtype)
        am = np.asarray(ex["attention_mask"])
        print("attention_mask:", am.shape, "sum=", int(am.sum()), "unique=", np.unique(am).tolist())
        print("text:", (ex['text'][:80] + "…") if len(ex['text'])>80 else ex['text'])
        print("domain/domain_label:", ex["domain"], ex["domain_label"])

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_num_threads(max(1, os.cpu_count()//2))
    main()
