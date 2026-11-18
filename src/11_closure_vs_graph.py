"""
Step 11 —— 数据一致性检验：**已解析闭包**与**直接依赖图**在同口径上的集合比对
       （评审意见 A-02 / B-03）

问题在哪
--------
本项目用两种不同的原料描述"一个包被多少东西依赖"：

  原料 ① `closures.parquet`  —— deps.dev 给每个种子包返回的**已解析传递闭包**。
                                它包含 DIRECT 与 INDIRECT 节点，且只覆盖 14,993 个种子
                                （"榜内"包）；榜外叶子没有闭包记录。
  原料 ② `edges.csv` + 反向可达 —— 直接依赖边图（15,718 节点），
                                可对**任意**节点算出"谁传递依赖我"。

如果直接拿 ① 的计数和 ② 的计数相减（本项目的 `t21_truncation.json` 原来就是这么做的），
那是**两个不同总体相减**：①只数种子，②数全部节点。由此得到的"图口径比闭包口径多/少 N"
里混着两种完全不同的东西：

    真差异 —— 两条数据流对"谁依赖谁"的判断不一致（这才是我们想量的）
    伪差异 —— 总体不同（14,993 vs 15,718）与叶子截断（榜外叶子没有出边）

本脚本把总体**对齐**：只在"图内种子"这一个共同总体 S 上，
对每个包 v 比较两个集合

    A(v) = { s ∈ S : v ∈ closure(s) }            （闭包口径）
    B(v) = { s ∈ S : s 在图中反向可达 v }         （图口径）

并报告 A∖B（闭包说有、图说没有 = 图漏边/漏层）与 B∖A（图说有、闭包说没有 =
声明层回退引入的边，或图把某条路径当成了直接依赖）。

为什么这两个集合**本来就不应该完全相等**（先说清楚，避免把正常差异当 bug）
--------------------------------------------------------------------------------
  * 榜外叶子（725 个）我们**没有抓它们的依赖**，所以"经过叶子的更深路径"在图里被截断
    → 这会系统性产生 A∖B。
  * 部分种子的闭包来自 PyPI `requires_dist` 回退（relation = DIRECT_DECLARED），
    它只有一层的声明，没有解析后的传递结构 → 也会产生偏差。
  * 闭包里的 INDIRECT 节点在同一次解析中确定；图里要靠"直接边 + 传递"复现，
    只要中间某一跳落在叶子截断区，就复现不出来。
因此本脚本的产出不是"对不对"，而是**偏差有多大、偏在谁身上** —— 这是可写进局限性的量。

产出
----
results/tables/t11_closure_vs_graph.csv      每个包一行：两种口径的集合大小、交并、Jaccard
results/tables/t11_closure_vs_graph_summary.json  总体一致性记账与 Top-20 分歧清单
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (abspath, bits_to_mask, CFG, ensure_parent, Log, mask_to_bits,
                    popcount_bits, rank_by, read_csv_safe, read_universe, write_json)

log = Log("11_closure_vs_graph")
TBL = abspath(CFG["tables"]["outdir"])


def main() -> None:
    nodes = read_csv_safe("data/processed/nodes.csv")
    cl = pd.read_parquet(abspath("data/processed/closures.parquet"))
    bits = np.load(abspath("data/processed/reach_bits.npy"))
    rnames = json.load(open(abspath("data/processed/reach_nodes.json"), encoding="utf-8"))
    pos = {v: i for i, v in enumerate(rnames)}
    n = len(rnames)

    # ---- 共同总体 S：图内的种子（榜内且 deps.dev 有记录）
    uni = read_universe()
    universe_seeds = set(uni["project"])
    S = [s for s in universe_seeds if s in pos]
    S_set = set(S)
    log(f"共同总体 S = 图内种子 {len(S)}（universe {len(universe_seeds)}，"
        f"图节点 {n}）")

    # ---- A(v)：闭包口径（只在 S 上数）
    cl = cl[cl["relation"] != "SELF"]
    cl = cl[cl["seed"].isin(S_set)]
    inv: dict[str, set[str]] = {}
    for seed, dep in zip(cl["seed"], cl["dep"]):
        if dep in pos:                    # 包必须在图里才有 B(v) 可比
            inv.setdefault(dep, set()).add(seed)
    log(f"闭包记录 {len(cl)} 行，覆盖 {len(inv)} 个图内包")

    # ---- B(v)：图口径（反向可达 ∩ S）
    # 全程用位集：先把"图内种子"打包成一个掩码，与每个包的反向可达位集求交，
    # 交集的大小就是 B(v)，而 A∩B 用**直接位测试**算（总工作量 = Σ|A(v)| ≈ 19 万次）。
    nw = bits.shape[1]
    seed_idx = np.array([pos[s] for s in S], dtype=np.int64)
    seed_mask_arr = np.zeros(n, dtype=bool)
    seed_mask_arr[seed_idx] = True
    seed_bits = mask_to_bits(seed_mask_arr, nw)
    inter = bits & seed_bits[None, :]
    nB_all = popcount_bits(inter)

    def _has(row: np.ndarray, j: int) -> bool:
        return bool((row[j >> 6] >> np.uint64(j & 63)) & np.uint64(1))

    rows = []
    for i, v in enumerate(rnames):
        A = inv.get(v, set())
        nA = len(A)
        nB = int(nB_all[i])
        nAB = 0
        if nA:
            row = inter[i]
            for s in A:
                if _has(row, pos[s]):
                    nAB += 1
        jac = nAB / (nA + nB - nAB) if (nA + nB - nAB) else 1.0
        rows.append({"name": v, "n_closure": nA, "n_graph": nB, "n_both": nAB,
                     "only_closure": nA - nAB, "only_graph": nB - nAB,
                     "jaccard": jac})

    df = pd.DataFrame(rows)
    df.to_csv(ensure_parent(TBL / "t11_closure_vs_graph.csv"), index=False, encoding="utf-8")

    # ---- 汇总
    both = df[(df["n_closure"] > 0) | (df["n_graph"] > 0)]
    gap = both["n_graph"] - both["n_closure"]
    abs_gap = gap.abs()
    jac_pos = both.loc[(both["n_closure"] > 0) & (both["n_graph"] > 0), "jaccard"]
    summary = {
        "common_population_S": len(S),
        "universe_seeds": len(universe_seeds),
        "graph_nodes": n,
        "closure_rows_used": int(len(cl)),
        "packages_in_graph": int(len(df)),
        "packages_with_any_dependents": int(len(both)),
        "packages_in_both_sources": int(((both["n_closure"] > 0) &
                                         (both["n_graph"] > 0)).sum()),
        "only_in_closure": int(((both["n_closure"] > 0) &
                                (both["n_graph"] == 0)).sum()),
        "only_in_graph": int(((both["n_closure"] == 0) &
                              (both["n_graph"] > 0)).sum()),
        # 集合级差异（严格同口径后仍然存在的真差异）= 评审 B-03 要求的 FP/FN
        "total_only_closure_pairs": int(both["only_closure"].sum()),
        "total_only_graph_pairs": int(both["only_graph"].sum()),
        "total_both_pairs": int(both["n_both"].sum()),
        # ★ 评审 B-03：有符号差与**绝对差**分开报。
        # 有符号差的 p95 不能冒充"95% 的节点误差在 ±x 以内"——大的负差不会进入上尾。
        "signed_gap_median": float(gap.median()) if len(both) else 0.0,
        "signed_gap_mean": float(gap.mean()) if len(both) else 0.0,
        "signed_gap_p05": float(np.percentile(gap, 5)) if len(both) else 0.0,
        "signed_gap_p95": float(np.percentile(gap, 95)) if len(both) else 0.0,
        "signed_gap_positive_share": float((gap > 0).mean()) if len(both) else 0.0,
        "signed_gap_negative_share": float((gap < 0).mean()) if len(both) else 0.0,
        "signed_gap_zero_share": float((gap == 0).mean()) if len(both) else 0.0,
        "abs_gap_median": float(abs_gap.median()) if len(both) else 0.0,
        "abs_gap_p90": float(np.percentile(abs_gap, 90)) if len(both) else 0.0,
        "abs_gap_p95": float(np.percentile(abs_gap, 95)) if len(both) else 0.0,
        "abs_gap_p99": float(np.percentile(abs_gap, 99)) if len(both) else 0.0,
        "abs_gap_max": float(abs_gap.max()) if len(both) else 0.0,
        "spearman_within_common_population": float(
            both["n_closure"].corr(both["n_graph"], method="spearman")) if len(both) else None,
        "jaccard_median": float(jac_pos.median()) if len(jac_pos) else None,
        "jaccard_p05": float(np.percentile(jac_pos, 5)) if len(jac_pos) else None,
        "jaccard_mean": float(jac_pos.mean()) if len(jac_pos) else None,
    }

    # ---- Top-20 分歧清单（按集合差异大小）
    both = both.copy()
    both["abs_gap"] = both["only_closure"] + both["only_graph"]
    top = both.sort_values(["abs_gap", "name"], ascending=[False, True]).head(20)
    summary["top20_disagreements"] = top[[
        "name", "n_closure", "n_graph", "n_both", "only_closure", "only_graph",
        "jaccard"]].to_dict(orient="records")

    # ---------------------------------------------------------- 敏感性（评审 A-02 第 5 条）
    # 用两个变体回答"主要榜单会不会被这些结构性偏差带偏"：
    #   ① 全部图内种子（主口径）
    #   ② 只保留"严格完整"的种子（把回退到声明层的种子排除）
    univ_cov = read_csv_safe("data/processed/universe_coverage.csv")
    if "closure_complete_strict" in univ_cov.columns:
        strict_seeds = set(univ_cov.loc[univ_cov["closure_complete_strict"]
                                        .astype(str).str.lower() == "true", "seed"])
    else:
        strict_seeds = set(univ_cov.loc[univ_cov["deps_status"] == "ok", "seed"])
    sens = {}
    for tag, seeds_sub in (("all_graph_seeds", S_set),
                           ("strict_complete_seeds_only", S_set & strict_seeds)):
        if not seeds_sub:
            continue
        inv2: dict[str, set[str]] = {}
        for seed, dep in zip(cl["seed"], cl["dep"]):
            if seed in seeds_sub and dep in pos:
                inv2.setdefault(dep, set()).add(seed)
        n_cl2 = np.array([len(inv2.get(x, ())) for x in rnames], dtype=np.int64)
        rho = pd.Series(n_cl2).corr(pd.Series(nB_all), method="spearman")
        a = rank_by({str(k): float(v) for k, v in zip(rnames, n_cl2)})[:20]
        b = rank_by({str(k): float(v) for k, v in zip(rnames, nB_all)})[:20]
        sens[tag] = {"n_seeds": len(seeds_sub),
                     "spearman_closure_vs_graph": float(rho) if rho == rho else None,
                     "sum_closure_side": int(n_cl2.sum()),
                     "sum_graph_side": int(nB_all.sum()),
                     "top20_overlap": len(set(a) & set(b))}
    summary["sensitivity"] = sens
    log("敏感性（A-02 第 5 条）:\n" + json.dumps(sens, ensure_ascii=False, indent=2))

    # 评审 A-04：榜外叶子是**未扩展边界节点**（我们没抓它们的出边），必须显式标注
    seeded = set(rnames) & S_set
    summary["boundary_nodes"] = {
        "n_seed_nodes": len(seeded),
        "n_unexpanded_leaf_nodes": int(len(rnames) - len(seeded)),
        "note": "榜外叶子没有抓到出边，因此经过它们的更深路径在图口径里被截断；"
                "它们在 nodes.csv 里以 in_seed=False 标注，不应被当成『无依赖』。",
    }

    write_json(ensure_parent(TBL / "t11_closure_vs_graph_summary.json"), summary)
    log("同口径一致性记账:\n" + json.dumps(
        {k: v for k, v in summary.items() if k != "top20_disagreements"},
        ensure_ascii=False, indent=2))
    log("Top-20 分歧:\n" + top.to_string(index=False))
    log("写出 t11_* 表")
    log.close()


if __name__ == "__main__":
    main()
