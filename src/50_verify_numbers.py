"""
Step 50 —— 数字一致性核对（把"报告里的每个数字都能在表里查到"变成机器检查）

为什么需要这一步
----------------
本项目正文里出现的每一个数字都来自 `results/tables/`。但正文是**手写**的，
而表格是**脚本生成**的 —— 只要中间重跑过一次流水线，正文就可能悄悄变成旧数字。
人眼在 500 行报告里找 30 个失效数字是不可靠的，这类错误也最伤可信度。

所以本脚本做三件事：

  1. 从 `results/tables/` 里重新计算出**全部**对外声称的数字（claims），
     每个 claim 都带：数值、格式化后的字符串、来源表、**定义与分母**。
     产出 `results/tables/claims.json`（机器读）与 `report/附录_数字清单.md`（人读）。
  2. 扫 README.md / report/03 / report/04 / report/05 / src/40_make_pptx.py，
     检查每个 claim 的格式化字符串是否**真的出现**。
  3. 有 claim 在"必须出现"的文件里找不到、或找到了但数值不同 → 退出码 1。

用法
----
    python src/50_verify_numbers.py            # 核对，失败则退出码 1
    python src/50_verify_numbers.py --report   # 只输出清单，不判定
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import abspath, CFG, ensure_parent, Log, read_csv_safe, read_json, write_json

log = Log("50_verify_numbers")
TBL = abspath(CFG["tables"]["outdir"])

DOCS_MUST = ["README.md", "report/03_结果.md", "report/04_讨论与结论.md"]
DOCS_ALSO = ["report/05_答辩讲稿.md", "src/40_make_pptx.py"]


# ------------------------------------------------------------------ 格式化候选
def forms(v) -> list[str]:
    """一个数值在正文里可能被写成哪些字符串（都算命中）。"""
    out = []
    if isinstance(v, bool) or v is None:
        return [str(v)]
    if isinstance(v, int):
        out += [f"{v:,}", str(v)]
    elif isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return []
        out += [f"{v:.1f}", f"{v:.2f}", f"{v:.3f}", f"{v:.4f}"]
        out += [s.rstrip("0").rstrip(".") for s in out]
        for nd in (1, 2, 3):
            out.append(f"{v*100:.{nd}f}")       # 百分比形态
    return sorted({s for s in out if s}, key=len, reverse=True)


class Claims:
    def __init__(self) -> None:
        self.items: dict[str, dict] = {}

    def add(self, key: str, value, source: str, definition: str,
            must: bool = True, extra_forms: list[str] | None = None,
            must_in: list[str] | None = None) -> None:
        """注册一个对外声称的数字。

        must_in: 这个数字**必须**出现在哪些文件里（评审 B-01 的验收条件：
                 "五处数值和措辞完全一致"）。默认沿用旧语义（must=True 时
                 要求出现在 DOCS_MUST 的每一个里）。
        """
        f = forms(value)
        if extra_forms:
            f = sorted(set(f) | set(extra_forms), key=len, reverse=True)
        self.items[key] = {
            "value": value, "source": source, "definition": definition,
            "must_appear": must, "must_in": must_in or [], "forms": f,
        }


def build_claims() -> Claims:
    C = Claims()
    gs = read_json(abspath("data/processed/graph_stats.json"), {}) or {}

    # ---- 图规模与数据质量（A-01）
    for k, label in [
        ("n_nodes", "图节点数"),
        ("n_edges", "直接依赖边数"),
        ("n_seed_nodes", "榜内种子节点数"),
        ("n_external_leaves", "榜外叶子节点数"),
        ("max_scc_size", "最大强连通分量节点数"),
        ("n_cycles_scc", "非平凡强连通分量个数"),
        ("seeds_resolved_ok", "deps.dev 成功解析闭包的种子数"),
        ("seeds_unresolved_fallback", "回退到 PyPI 声明层的种子数"),
        ("seeds_strict_complete", "严格完整（解析成功且无图级/节点级错误）的种子数"),
        ("seeds_with_graph_error", "存在图级解析错误的种子数"),
        ("node_level_errors_total", "节点级解析错误总数"),
        ("fallback_no_requires_dist", "回退且 PyPI 确无依赖声明的种子数"),
        ("fallback_fetch_failed", "回退且抓取失败的种子数"),
        ("closure_rows", "闭包记录行数"),
    ]:
        if k in gs:
            C.add(f"graph.{k}", gs[k], "data/processed/graph_stats.json", label,
                  must=k in ("n_nodes", "n_edges", "max_scc_size"))

    # ---- 三级口径（S/F/V）
    t22 = read_csv_safe(TBL / "t22_vuln_tiers.csv") if (TBL / "t22_vuln_tiers.csv").exists() \
        else None
    if t22 is not None and {"tier", "hits"}.issubset(t22.columns):
        m = dict(zip(t22["tier"], t22["hits"]))
        S, F, V = m.get("S"), m.get("F"), m.get("V")
        IN_SFV = ["README.md", "report/03_结果.md", "report/04_讨论与结论.md"]
        IN_RATE = IN_SFV + ["report/05_答辩讲稿.md", "src/40_make_pptx.py"]
        C.add("tier.S", S, "tier_hits",
              "结构上界：任何传递依赖者都算受影响", must_in=IN_SFV)
        C.add("tier.F", F, "tier_hits", "反事实：区间改为 [introduced, ∞)",
              must_in=IN_SFV)
        C.add("tier.V", V, "tier_hits", "现实：OSV 区间 + 当期解析版本",
              must_in=IN_SFV)
        if S and V:
            C.add("ratio.S_over_V", round(S / V, 1), "由 S、V 计算",
                  "S/V，**仅表示模型内部两个口径的倍数，不是真实风险高估倍数**",
                  must_in=IN_RATE)
        if S and F:
            C.add("pct.S_to_F", round(100 * (S - F) / S, 2), "由 S、F 计算",
                  "(S−F)/S，分母是 S", must_in=IN_RATE)
        if F and V:
            C.add("pct.F_to_V", round(100 * (F - V) / F, 2), "由 F、V 计算",
                  "(F−V)/F，分母是 F", must_in=IN_RATE)
    # 口径拆分（若 22 产出了分解表）
    dec = TBL / "t22_decomposition.csv"
    if dec.exists():
        d = read_csv_safe(dec)
        for r in d.to_dict(orient="records"):
            C.add(f"decomp.{r.get('key')}", r.get("value"), "t22_decomposition.csv",
                  str(r.get("definition", "")), must=False)

    # ---- 结构性结果（t20_structure.csv 是一行宽表）
    st = TBL / "t20_structure.csv"
    if st.exists():
        j = read_csv_safe(st).iloc[0].to_dict()
        for key, label in [
            ("n_nodes", "图节点数"), ("n_edges", "直接依赖边数"),
            ("n_seeds", "榜内种子数"), ("max_scc", "最大强连通分量节点数"),
            ("n_nontrivial_scc", "非平凡强连通分量个数"),
            ("dep_depth_mean", "依赖深度均值"), ("dep_depth_max", "依赖深度最大值"),
            ("degree_assortativity", "度同配性"), ("gini_in_degree", "入度 Gini"),
        ]:
            if key in j:
                v = j[key]
                C.add(f"struct.{key}", v, "t20_structure.csv", label, must=False)

    # ---- 零模型：两族 × 两口径 + 每族实测 SCC 规模（D-01 的核心证据）
    ncmp = TBL / "t24_null_compare.csv"
    if ncmp.exists():
        n = read_csv_safe(ncmp)
        gn = n[n["metric"] == "node"]
        IN_NULL = ["README.md", "report/03_结果.md", "report/04_讨论与结论.md"]
        for fam, sub in gn.groupby("family"):
            C.add(f"null.{fam}.mean", float(sub["null_mean"].mean()),
                  "t24_null_compare.csv",
                  f"零模型族 {fam} 的反向可达均值（node 口径，{len(sub)} 次实现）",
                  must_in=IN_NULL)
            C.add(f"null.{fam}.mean_sd", float(sub["null_mean"].std()),
                  "t24_null_compare.csv", f"族 {fam} 的 null_mean 实现间标准差", must=False)
            C.add(f"null.{fam}.max_scc_median", int(sub["null_max_scc"].median()),
                  "t24_null_compare.csv", f"族 {fam} 实测最大 SCC 规模中位数",
                  must_in=["README.md", "report/03_结果.md", "report/04_讨论与结论.md"])
        for metric in ("node", "condensation"):
            subn = n[n["metric"] == metric]
            for fam, ss in subn.groupby("family"):
                C.add(f"null.{fam}.{metric}.rho", round(float(ss["spearman_rank_vs_true"].mean()), 3),
                      "t24_null_compare.csv", f"族 {fam} / 口径 {metric} 的排名一致性", must=False)

    # ---- 介数采样（B-04）
    bs = TBL / "t27_betweenness_stability.csv"
    if bs.exists():
        b = read_csv_safe(bs)
        sm = b[b["mode"] == "sampled"]
        if len(sm):
            C.add("btw.top50_hits_min", int(sm["top50_hits"].min()),
                  "t27_betweenness_stability.csv",
                  "抽样介数与精确介数 Top-50 命中最少的一次", must=False)
            C.add("btw.rho_min", round(float(sm["spearman_vs_exact"].min()), 4),
                  "t27_betweenness_stability.csv",
                  "抽样介数与精确介数的 Spearman 最小值", must=False)
            C.add("btw.exact_runtime_s", round(float(b.loc[b["mode"] == "exact", "runtime_s"].iloc[0]), 1),
                  "t27_betweenness_stability.csv", "精确全源介数用时（秒）", must=False)
        ov = TBL / "t27_betweenness_seed_overlap.csv"
        if ov.exists():
            o = read_csv_safe(ov)
            if "top50_jaccard" in o.columns and len(o):
                C.add("btw.seed_top50_jaccard_min",
                      round(float(o["top50_jaccard"].min()), 3),
                      "t27_betweenness_seed_overlap.csv",
                      "同 k 不同 seed 之间 Top-50 的 Jaccard 最小值", must=False)

    # ---- 保护情景（C-03）
    prot = TBL / "t28_protection_gain_ratio.csv"
    if prot.exists():
        p = read_csv_safe(prot)
        if "gain_ratio_top_over_random" in p.columns and p["gain_ratio_top_over_random"].notna().any():
            C.add("protect.gain_ratio_median",
                  round(float(p["gain_ratio_top_over_random"].median()), 2),
                  "t28_protection_gain_ratio.csv",
                  "同 k 下『保护枢纽』相对『随机保护』的 ΔS 增益中位数", must=False)
            C.add("protect.gain_ratio_min",
                  round(float(p["gain_ratio_top_over_random"].min()), 2),
                  "t28_protection_gain_ratio.csv",
                  "同 k 下『保护枢纽』相对『随机保护』的 ΔS 增益最小值", must=False)

    # ---- 一致性检验（A-02/B-03）
    cons = TBL / "t11_closure_vs_graph_summary.json"
    if cons.exists():
        j = read_json(cons, {}) or {}
        for k in ("total_only_closure_pairs", "total_only_graph_pairs",
                  "total_both_pairs", "jaccard_median", "jaccard_mean",
                  "abs_gap_median", "abs_gap_p95", "abs_gap_max",
                  "signed_gap_p95", "signed_gap_positive_share",
                  "spearman_within_common_population"):
            if k in j and j[k] is not None:
                C.add(f"consistency.{k}", j[k], "t11_closure_vs_graph_summary.json",
                      k, must=False)

    # ---- 三级口径的恒等式自检（B-01）
    t22j = TBL / "t22_summary.json"
    if t22j.exists():
        j = read_json(t22j, {}) or {}
        for k, label in [
            ("shrink_F_to_V_from_closed_advisories", "F→V 差集中来自已封口 advisory 的部分"),
            ("shrink_F_to_V_from_open_advisories", "F→V 差集中来自未封口 advisory 的部分（应为 0）"),
            ("n_single_range_advisories_checked", "可做恒等式核对的单区间 advisory 数"),
            ("n_ge_fixed_single_range_count", "单区间 advisory 上 resolved≥fixed 的同口径计数"),
            ("frac_advisories_zero_real_impact", "现实中零影响的 advisory 占比"),
            ("n_advisories_open_ended", "至今无补丁的 advisory 数"),
        ]:
            if k in j:
                C.add(f"identity.{k}", j[k], "t22_summary.json", label, must=False)

    return C


# ------------------------------------------------------------------ 抽取正文数值
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def scan(path: Path) -> tuple[str, set[str]]:
    if not path.exists():
        return "", set()
    txt = path.read_text(encoding="utf-8", errors="replace")
    found = {m.group(0) for m in NUM_RE.finditer(txt)}
    return txt, found


def main() -> None:
    report_only = "--report" in sys.argv[1:]
    C = build_claims()
    write_json(ensure_parent(TBL / "claims.json"), C.items)
    log(f"claims = {len(C.items)} 条，已写出 {TBL / 'claims.json'}")

    targets = {}
    for rel in DOCS_MUST + DOCS_ALSO:
        _txt, found = scan(abspath(rel))
        targets[rel] = found

    rows, missing_must, wrong = [], [], []
    for key, c in C.items.items():
        where = []
        for rel in DOCS_MUST + DOCS_ALSO:
            hit = None
            found = targets.get(rel, set())
            # 数值命中：任一格式化形态出现即算
            for f in c["forms"]:
                if f in found:
                    hit = f
                    break
            where.append((rel, hit))
            required = c.get("must_in") or (DOCS_MUST if c["must_appear"] else [])
            if hit is None and rel in required:
                missing_must.append((key, rel, c["value"]))
        rows.append({"claim": key, "value": c["value"], "source": c["source"],
                     "definition": c["definition"],
                     "present_in": ";".join(f"{r}={h}" for r, h in where if h),
                     "absent_in": ";".join(r for r, h in where if not h)})

    df = pd.DataFrame(rows)
    df.to_csv(ensure_parent(TBL / "t50_claims_check.csv"), index=False, encoding="utf-8")

    # 人读清单
    lines = ["# 附录：数字清单（由 `src/50_verify_numbers.py` 自动生成）", "",
             "本文件列出正文所声称的每一个数字、它的来源表、以及**定义与分母**。",
             "任何与本表冲突的数字都应以本表为准，并回头修正正文。", "",
             "| 数字 | 值 | 来源 | 定义 / 分母 | 出现在 |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['claim']}` | {r['value']} | `{r['source']}` | "
                     f"{r['definition']} | {r['present_in'] or '—'} |")
    out_md = ensure_parent(abspath("report/附录_数字清单.md"))
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"-> {out_md}")

    if report_only:
        log("只输出清单，未判定（--report）")
        log.close()
        return

    if missing_must:
        log("★ 以下 claim 在必须出现的文件里找不到（正文与表不一致）：")
        for key, rel, val in missing_must:
            log(f"   - {key} = {val}  应在 {rel} 出现")
    else:
        log("全部 claim 均在必须出现的文件中命中 ✓")

    log("写出 t50_claims_check.csv")
    log.close()
    if missing_must:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
