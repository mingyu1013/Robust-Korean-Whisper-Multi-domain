# asr_normalize_metrics.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import unicodedata as ud
import regex as re
from typing import List, Tuple, Dict, Optional, Any

# ----------------------------
# 0) 공통 유틸
# ----------------------------
_PUNCT_CATS = {"P", "S"}  # Punctuation, Symbol

def _to_lower_if_ascii(s: str) -> str:
    # 영문 대소만 소문자화 (한글에는 영향 없음)
    return "".join(ch.lower() if "LATIN" in ud.name(ch, "") else ch for ch in s)

def _strip_punct_symbols(s: str) -> str:
    # 카테고리가 P(문장부호) 또는 S(기호)이면 제거
    out = []
    for ch in s:
        cat0 = ud.category(ch)[0]  # e.g., 'P', 'S', 'L'...
        if cat0 in _PUNCT_CATS:
            continue
        out.append(ch)
    return "".join(out)

def _squash_spaces(s: str) -> str:
    # 연속 공백을 하나로, 앞뒤 공백 제거
    return re.sub(r"\s+", " ", s).strip()

def normalize_common(text: str) -> str:
    """
    공통 정규화:
      1) 유니코드 정준화(NFKC)
      2) 영문만 소문자화
      3) 특수기호/문장부호 제거 (P/S 카테고리)
      4) 공백 정리
    """
    if text is None:
        text = ""
    s = ud.normalize("NFKC", str(text))
    s = _to_lower_if_ascii(s)
    s = _strip_punct_symbols(s)
    s = _squash_spaces(s)
    return s

# ----------------------------
# 1) 토크나이저들
# ----------------------------
# 1-1) WER: 공백 기준
def tokenize_wer(s: str) -> List[str]:
    s = normalize_common(s)
    if not s:
        return []
    return s.split(" ")

# 1-2) CER: 문자 단위 (옵션: 스페이스 포함 여부)
def tokenize_cer(s: str, include_space: bool = True) -> List[str]:
    s = normalize_common(s)
    if not s:
        return []
    if include_space:
        return list(s)
    else:
        return list(s.replace(" ", ""))

# 1-3) PER: 한국어는 음절→자모(초/중/종성) 분해, 그 외는 문자 단위
#  - 공백은 무시(논문 설정)
_JAMO_BASE = {
    "CHO": 0x1100,  # 초성
    "JUNG": 0x1161, # 중성
    "JONG": 0x11A7, # 종성(0x11A8부터 실제 자모)
}
_CHO_LIST = [chr(c) for c in range(0x1100, 0x1113)]
_JUNG_LIST = [chr(c) for c in range(0x1161, 0x1176)]
_JONG_LIST = [chr(c) for c in range(0x11A8, 0x11C3)]

def _decompose_hangul_jamo(ch: str) -> List[str]:
    """한글 음절(가~힣)을 초/중/종성 자모로 분해."""
    code = ord(ch)
    S_BASE, L_BASE, V_BASE, T_BASE = 0xAC00, 0x1100, 0x1161, 0x11A7
    L_COUNT, V_COUNT, T_COUNT = 19, 21, 28
    N_COUNT = V_COUNT * T_COUNT
    S_COUNT = L_COUNT * N_COUNT

    if 0xAC00 <= code < 0xAC00 + S_COUNT:
        S_INDEX = code - S_BASE
        L = L_BASE + S_INDEX // N_COUNT
        V = V_BASE + (S_INDEX % N_COUNT) // T_COUNT
        T = T_BASE + (S_INDEX % T_COUNT)
        out = [chr(L), chr(V)]
        if T != T_BASE:
            out.append(chr(T))
        return out
    else:
        return [ch]

def tokenize_per(s: str) -> List[str]:
    s = normalize_common(s)
    if not s:
        return []
    tokens: List[str] = []
    for ch in s:
        if ch == " ":
            # PER에서는 공백 무시
            continue
        name = ud.name(ch, "")
        if "HANGUL SYLLABLE" in name:
            tokens.extend(_decompose_hangul_jamo(ch))
        else:
            # 비한글: 문자 단위 대체(실제 음소가 아님을 주의)
            tokens.append(ch)
    return tokens

# 1-4) TER: BPE 토큰. tiktoken 있으면 사용, 없으면 폴백.
_TIKTOKEN = None
def _maybe_load_tiktoken():
    global _TIKTOKEN
    if _TIKTOKEN is None:
        try:
            import tiktoken  # type: ignore
            _TIKTOKEN = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN = False
    return _TIKTOKEN

def tokenize_ter(s: str) -> List[str]:
    s = normalize_common(s)
    if not s:
        return []
    enc = _maybe_load_tiktoken()
    if enc:
        ids = enc.encode(s)
        return [f"▁{i}" for i in ids]
    # 폴백: 대충 서브워드 흉내
    out: List[str] = []
    for w in s.split(" "):
        if len(w) <= 4:
            out.append(w)
        else:
            head = w[:4]
            out.append(head)
            tail = w[4:]
            out.extend([tail[i:i+2] for i in range(0, len(tail), 2)])
    return out

# ----------------------------
# 2) 편집거리(정렬)와 ER
# ----------------------------
def _edit_ops(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """
    반환: (S, D, I, N)  N은 ref 길이
    """
    N, M = len(ref), len(hyp)
    dp = [[0]*(M+1) for _ in range(N+1)]
    bt = [[None]*(M+1) for _ in range(N+1)]
    for i in range(1, N+1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(1, M+1):
        dp[0][j] = j
        bt[0][j] = "I"
    for i in range(1, N+1):
        for j in range(1, M+1):
            cost_sub = dp[i-1][j-1] + (0 if ref[i-1] == hyp[j-1] else 1)
            cost_del = dp[i-1][j] + 1
            cost_ins = dp[i][j-1] + 1
            best = min(cost_sub, cost_del, cost_ins)
            dp[i][j] = best
            if best == cost_sub:
                bt[i][j] = "M" if ref[i-1] == hyp[j-1] else "S"
            elif best == cost_del:
                bt[i][j] = "D"
            else:
                bt[i][j] = "I"
    i, j = N, M
    S = D = I = 0
    while i > 0 or j > 0:
        tag = bt[i][j]
        if tag == "M":
            i -= 1; j -= 1
        elif tag == "S":
            S += 1; i -= 1; j -= 1
        elif tag == "D":
            D += 1; i -= 1
        elif tag == "I":
            I += 1; j -= 1
        else:
            break
    return S, D, I, N

def _er_from_ops(S: int, D: int, I: int, N: int) -> float:
    if N == 0:
        return 0.0
    return (S + D + I) / float(N)

# ----------------------------
# 3) 메트릭 계산 API
# ----------------------------
def _pack(prefix: str, S:int, D:int, I:int, N:int, err_key:str, err_val:float):
    return {
        f"{prefix}S": S, f"{prefix}D": D, f"{prefix}I": I, f"{prefix}N": N,
        err_key: err_val
    }

def compute_wer_counts(ref: str, hyp: str):
    r, h = tokenize_wer(ref), tokenize_wer(hyp)
    S,D,I,N = _edit_ops(r,h)
    return _pack("w_", S,D,I,N, "WER", _er_from_ops(S,D,I,N))

def compute_cer_counts(ref: str, hyp: str, include_space: bool = False):
    r, h = tokenize_cer(ref, include_space), tokenize_cer(hyp, include_space)
    S,D,I,N = _edit_ops(r,h)
    key = "CER_sp" if include_space else "CER"
    pre = "csp_" if include_space else "c_"
    return _pack(pre, S,D,I,N, key, _er_from_ops(S,D,I,N))

def compute_per_counts(ref: str, hyp: str):
    r, h = tokenize_per(ref), tokenize_per(hyp)
    S,D,I,N = _edit_ops(r,h)
    return _pack("p_", S,D,I,N, "PER", _er_from_ops(S,D,I,N))

def compute_ter_counts(ref: str, hyp: str):
    r, h = tokenize_ter(ref), tokenize_ter(hyp)
    S,D,I,N = _edit_ops(r,h)
    return _pack("t_", S,D,I,N, "TER", _er_from_ops(S,D,I,N))

# ----------------------------
# 4) 배치 헬퍼 (네 JSON 구조용)
# ----------------------------
def evaluate_item(item: Dict[str, Any], pred_key: str = "text_pred_mas") -> Dict[str, Any]:
    """
    item: {
      "id": ..., "domain": ..., "text_ref": ..., "text_ref_norm": (opt),
      pred_key: ...
    }
    """
    ref_raw = item.get("text_ref", "") or ""
    # 정규화본이 있으면 우선 사용, 없으면 동일 규칙으로 정규화
    ref = item.get("text_ref_norm") or normalize_common(ref_raw)
    hyp = item.get(pred_key, "") or ""

    out: Dict[str, Any] = {"id": item.get("id"), "domain": item.get("domain")}

    # 지표별 SDIN + 에러율
    w = compute_wer_counts(ref, hyp)                         # w_S/D/I/N, WER
    csp = compute_cer_counts(ref, hyp, include_space=True)   # csp_S/D/I/N, CER_sp
    c   = compute_cer_counts(ref, hyp, include_space=False)  # c_S/D/I/N,  CER
    p   = compute_per_counts(ref, hyp)                       # p_S/D/I/N,  PER
    t   = compute_ter_counts(ref, hyp)                       # t_S/D/I/N,  TER

    out.update(w); out.update(csp); out.update(c); out.update(p); out.update(t)

    # 보고 편의: alias는 **WER 기준 S/D/I/N**으로 제공
    out["S"] = out["w_S"]; out["D"] = out["w_D"]; out["I"] = out["w_I"]; out["N"] = out["w_N"]

    return out

def evaluate_list(items: List[Dict[str, Any]], pred_key: str = "text_pred_mas") -> List[Dict[str, Any]]:
    return [evaluate_item(it, pred_key=pred_key) for it in items]
