# eval_from_json_macro.py
# -*- coding: utf-8 -*-
"""
ASR JSON 평가
- 통합 결과 = 도메인별 평균의 산술평균(매크로 평균)
- per-item, per-domain 결과도 함께 출력/저장
"""

import os, json, argparse, glob
from statistics import mean, median
from typing import List, Dict, Any, Tuple
import pandas as pd

# ====== 외부 모듈: item 단위 정규화 지표 계산기 ======
from asr_normalize_metrics import evaluate_list  # (ref,pred)->WER/CER/... 계산

# ----------------------------
# 공통 저장 플래그
# ----------------------------
def add_common_save_flag(parser, default_save='off'):
    parser.add_argument('--save', choices=['on', 'off'], default=default_save)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--run_name', type=str, default='run')
    return parser

def should_save(args) -> bool:
    return getattr(args, 'save', 'off') == 'on'

# ----------------------------
# 로딩 유틸
# ----------------------------
def _load_json_any(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read().strip()
    if not txt:
        return []
    if txt[0] != '[':
        # JSON Lines
        return [json.loads(line) for line in txt.splitlines() if line.strip()]
    return json.loads(txt)

def _collect_inputs(args) -> List[str]:
    paths: List[str] = []
    if args.json:
        paths.append(args.json)
    if args.json_glob:
        paths.extend(sorted(glob.glob(args.json_glob)))
    # 중복 제거
    uniq, seen = [], set()
    for p in paths:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

# ----------------------------
# 평균/중앙 유틸
# ----------------------------
def _safe_mean(vals: List[float]) -> float:
    return float(mean(vals)) if vals else 0.0

def _safe_median(vals: List[float]) -> float:
    return float(median(vals)) if vals else 0.0

# ----------------------------
# 요약 집계
# ----------------------------
def summarize_per_item(items_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """per-item에서 직접 평균낸 ‘마이크로’ 요약. 참고용."""
    keys = []
    for d in items_metrics:
        for k in ("WER","CER_sp","CER","PER","TER"):
            if k in d: keys.append(k)
    keys = sorted(set(keys))
    out = {}
    for k in keys:
        vals = [float(x[k]) for x in items_metrics if k in x]
        out[k] = {"mean": _safe_mean(vals), "median": _safe_median(vals), "count": len(vals)}
    for k in ("S","D","I","N","w_S","w_D","w_I","w_N",
              "csp_S","csp_D","csp_I","csp_N",
              "c_S","c_D","c_I","c_N",
              "p_S","p_D","p_I","p_N",
              "t_S","t_D","t_I","t_N"):
        if any(k in x for x in items_metrics):
            vals = [int(x.get(k,0)) for x in items_metrics]
            out[k] = {"mean": _safe_mean(vals), "median": _safe_median(vals), "sum": int(sum(vals)), "count": len(vals)}
    return out

def split_by_domain(items_metrics: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for d in items_metrics:
        dom = d.get("domain", "UNKNOWN")
        groups.setdefault(str(dom), []).append(d)
    return groups

def summarize_per_domain(groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """각 도메인의 per-item 평균."""
    dom_summary: Dict[str, Dict[str, Any]] = {}
    for dom, rows in groups.items():
        dom_summary[dom] = summarize_per_item(rows)
    return dom_summary

def macro_from_domain(dom_summary: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    통합 = 도메인별 평균의 산술평균(매크로).
    각 지표에 대해 dom_summary[dom][metric]['mean']를 평균.
    """
    metrics = set()
    for dom, summ in dom_summary.items():
        for k, v in summ.items():
            if isinstance(v, dict) and "mean" in v:
                metrics.add(k)
    metrics = sorted(metrics)
    out = {"domains": 0}
    acc = {k: 0.0 for k in metrics}
    cnt = 0
    for dom, summ in dom_summary.items():
        has_any = False
        for k in metrics:
            if k in summ and "mean" in summ[k]:
                acc[k] += float(summ[k]["mean"])
                has_any = True
        if has_any:
            cnt += 1
    out["domains"] = cnt
    for k in metrics:
        out[k] = (acc[k] / cnt) if cnt > 0 else 0.0
    return out

# ----------------------------
# 메인
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="ASR JSON 평가(통합=도메인 매크로 평균)")
    add_common_save_flag(parser, default_save='on')
    parser.add_argument('--json', type=str, default=None, help='단일 JSON 경로')
    parser.add_argument('--json_glob', type=str, default=None, help='글롭 패턴')
    parser.add_argument('--pred_key', type=str, default='text_pred_mas_lora')
    parser.add_argument('--by_domain', action='store_true', help='도메인별 표 저장')
    parser.add_argument('--print_top_k', type=int, default=0)
    args = parser.parse_args()

    files = _collect_inputs(args)
    if not files:
        raise SystemExit("입력 JSON이 없습니다.")

    all_item_rows: List[Dict[str, Any]] = []
    for fp in files:
        data = _load_json_any(fp)
        if not isinstance(data, list):
            print(f"[WARN] {fp}: 최상위가 리스트가 아님. 건너뜀.")
            continue
        # 필드 보강
        for it in data:
            it.setdefault("id", None)
            it.setdefault("domain", None)
            it.setdefault("text_ref", it.get("text", ""))  # 호환
            it.setdefault(args.pred_key, it.get(args.pred_key, it.get("text_pred_mas_lora","")))
        rows = evaluate_list(data, pred_key=args.pred_key)
        for i, m in enumerate(rows):
            m["__file__"] = os.path.basename(fp)
            m.setdefault("id", data[i].get("id"))
            m.setdefault("domain", data[i].get("domain"))
        all_item_rows.extend(rows)

    if not all_item_rows:
        raise SystemExit("평가 가능한 항목이 없습니다.")

    # Per-item 미리보기
    df = pd.DataFrame(all_item_rows)
    cols_show = [c for c in ["__file__","id","domain","WER","CER_sp","CER","PER","TER"] if c in df.columns]
    print("\n=== Per-item metrics (head) ===")
    print(df[cols_show].head(20).to_string(index=False))

    # 도메인별 집계
    groups = split_by_domain(all_item_rows)
    dom_summary = summarize_per_domain(groups)

    # 통합 = 매크로 평균
    overall_macro = macro_from_domain(dom_summary)
    print("\n=== Overall (macro over domains) ===")
    print(f"domains={overall_macro['domains']}", end="")
    for k, v in overall_macro.items():
        if k == "domains": continue
        print(f"  {k}={v:.6f}", end="")
    print("")

    # 옵션: 도메인별 요약 출력
    if args.by_domain:
        print("\n=== By-domain means ===")
        for dom in sorted(dom_summary.keys()):
            summ = dom_summary[dom]
            line = [f"[{dom}]"]
            for k in ("WER","CER_sp","CER","PER","TER"):
                if k in summ and "mean" in summ[k]:
                    line.append(f"{k}={summ[k]['mean']:.6f}")
            print("  " + "  ".join(line))

    # 상위 샘플
    if args.print_top_k and "WER" in df.columns:
        print(f"\n=== Hardest samples by WER (top {args.print_top_k}) ===")
        topk = df.sort_values("WER", ascending=False).head(args.print_top_k)
        for _, r in topk.iterrows():
            print(f"- file={r.get('__file__')}  id={r.get('id')}  domain={r.get('domain')}  WER={r.get('WER'):.4f}")

    # 저장
    if should_save(args):
        os.makedirs(args.output_dir, exist_ok=True)
        per_item_csv = os.path.join(args.output_dir, f"{args.run_name}_per_item.csv")
        df.to_csv(per_item_csv, index=False, encoding='utf-8-sig')

        summary_json = {
            "overall_macro": overall_macro,
            "by_domain": dom_summary if args.by_domain else None,
            "count_items": int(len(all_item_rows)),
            "source_files": [os.path.basename(x) for x in files],
            "pred_key": args.pred_key,
        }
        summary_path = os.path.join(args.output_dir, f"{args.run_name}_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_json, f, ensure_ascii=False, indent=2)

        if args.by_domain:
            rows: List[Dict[str, Any]] = []
            for dom, summ in dom_summary.items():
                row = {"domain": dom}
                for k in ("WER","CER_sp","CER","PER","TER"):
                    if k in summ:
                        row[f"{k}_mean"] = summ[k]["mean"]
                        row[f"{k}_median"] = summ[k]["median"]
                        row[f"{k}_count"] = summ[k]["count"]
                rows.append(row)
            dom_csv = os.path.join(args.output_dir, f"{args.run_name}_by_domain.csv")
            pd.DataFrame(rows).to_csv(dom_csv, index=False, encoding="utf-8-sig")

        print(f"\n[Saved] per-item: {per_item_csv}")
        print(f"[Saved] summary : {summary_path}")
        if args.by_domain:
            print(f"[Saved] by-domain: {dom_csv}")

if __name__ == "__main__":
    main()
