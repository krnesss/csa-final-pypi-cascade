"""
Step 25 —— 验证实验：便宜的静态指标能不能**排序**真实级联影响面？

★ 措辞限定（评审意见 B-05）
---------------------------
本脚本早期版本把这三件事都叫"预测"。这不准确，必须收紧：

  * 检验 1 是**同时性**的：结构指标与"版本感知受影响包数"来自**同一份快照**，
    它衡量的是"便宜的指标能否替代昂贵的计算去排序"，**不是预测未来**。
  * 检验 2 才带一点时间先后（用当期网络去看更晚披露的漏洞），
    但仍然只是**分组关联**：网络本身是当期快照，不是"披露时点的网络"，
    因此不能说"用历史网络预测了未来"。
  * 检验 3 是**同期相关**：结构瓶颈 vs 历史 advisory 条数。

所以正文里一律写成"排序能力（rank correlation）/ 分组的关联强度"，
只有明确标注"这是同时性关联，不是因果与预测"时才允许出现"预测"二字。

要排除的替代解释（课程要求"结果是否超出证据"）
------------------------------------------------
我们宣称"反向 PageRank / 传递依赖者数 是好的瓶颈分数"。
一种反驳是：这些指标只是**重述**了我们要算的东西，所以当然相关。本脚本做三类**非循环**检验：

检验 1（软信息检验）：只用**最便宜**的指标（入度 = 直接依赖者数，看一眼就能算）
    能否排序**最难算**的口径（版本约束感知的真实受影响包数）？
    → 若 Spearman ρ 很高，说明工程上不必跑全量解析就能排序。

检验 2（时间分组检验）：把 OSV advisory 按发布年份切成 ≤T 与 >T 两部分，
    比较当期结构指标在两个分组上的秩相关是否相近。
    → 若两侧 ρ 相近，说明结论对时间分组不敏感（固定切片假设站得住）。
    **注意：这是分组关联，不是时间外推预测。**

检验 3（独立事件维度）：结构瓶颈分数 vs 该包在历史上**实际收到多少条 advisory**
    （一个完全独立于拓扑的"出事频率"）。两者若正相关，说明
    "结构上关键"与"历史上多灾"确实重合，是真实的双重风险；
    若无关，则说明"关键但不常出事"，这本身也是有用结论。

产出
----
results/tables/t25_predictive_power.csv
results/tables/t25_temporal_holdout.csv
results/tables/t25_incident_frequency.csv
results/tables/t25_wording_guard.json      本脚本使用的措辞规则（供 50_verify_numbers 交叉检查）
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import abspath, CFG, ensure_parent, Log, read_csv_safe

# bootstrap 里偶尔抽到常数数组（例如某子集全为同一个值），scipy 会发
# ConstantInputWarning 并返回 nan —— 这是预期内的，统计上按"该次抽样无信息"处理，
# 不影响点估计与 CI 的分位数。这里显式静音并把 nan 过滤掉（见 boot_spearman）。
warnings.filterwarnings("ignore", message="An input array is constant")

log = Log("25_exp_validation")
TBL = abspath(CFG["tables"]["outdir"])


def boot_spearman(x: np.ndarray, y: np.ndarray, n=1000, seed=12345) -> tuple[float, float, float]:
    """Spearman ρ 与 bootstrap 95% CI（课程要求"报告均值与波动"）。"""
    rho = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    n_ = len(x)
    vals = []
    for _ in range(n):
        i = rng.integers(0, n_, size=n_)
        try:
            vals.append(stats.spearmanr(x[i], y[i]).statistic)
        except Exception:  # noqa: BLE001
            continue
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return rho, np.nan, np.nan
    return rho, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    nm = pd.read_parquet(abspath("data/processed/node_metrics.parquet")).set_index("name")
    imp = read_csv_safe("results/tables/t22_advisory_impact.csv")
    osv = read_csv_safe("data/processed/osv_advisories.csv")

    # ---------------------------------------------------------- 检验 1：便宜指标 -> 真实影响面
    # 目标 y：每条 advisory 的版本感知受影响包数
    y_df = imp[imp["a_struct"] > 0].copy()
    y_df["y"] = y_df["a_version"]
    log(f"检验 1 样本：{len(y_df)} 条 advisory（有依赖者）")

    rows = []
    preds = ["in_degree", "n_transitive_dependents", "pagerank_importance",
             "betweenness", "kcore", "downloads", "cascade_depth"]
    for p in preds:
        x = y_df["package"].map(nm[p]).astype(float).to_numpy()
        y = y_df["y"].astype(float).to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        rho, lo, hi = boot_spearman(x[ok], y[ok])
        rows.append({"predictor": p, "n": int(ok.sum()), "spearman_rho": rho,
                     "ci95_low": lo, "ci95_high": hi})
    pdf = pd.DataFrame(rows).sort_values("spearman_rho", ascending=False)
    pdf.to_csv(ensure_parent(TBL / "t25_predictive_power.csv"), index=False, encoding="utf-8")
    log("检验 1（便宜指标对『版本感知受影响包数』的排序能力，同时性）:\n"
        + pdf.round(4).to_string(index=False))

    # 只用 a_version>0 的子集（对"完全没影响"的 case 不奖励"预测 0"）
    sub = y_df[y_df["y"] > 0]
    rows2 = []
    for p in preds:
        x = sub["package"].map(nm[p]).astype(float).to_numpy()
        yv = sub["y"].astype(float).to_numpy()
        ok = np.isfinite(x) & np.isfinite(yv)
        rho, lo, hi = boot_spearman(x[ok], yv[ok])
        rows2.append({"predictor": p, "n": int(ok.sum()), "spearman_rho": rho,
                      "ci95_low": lo, "ci95_high": hi, "subset": "a_version>0"})
    pd.DataFrame(rows2).sort_values("spearman_rho", ascending=False).to_csv(
        ensure_parent(TBL / "t25_predictive_power_strict.csv"), index=False, encoding="utf-8")
    log("检验 1b（仅 a_version>0 子集）:\n" +
        pd.DataFrame(rows2).sort_values("spearman_rho", ascending=False).round(4).to_string(index=False))

    # ---------------------------------------------------------- 检验 2：时间外推
    y_df["year"] = pd.to_datetime(y_df["published"], errors="coerce", utc=True).dt.year
    log("advisory 年份分布:\n" + y_df["year"].value_counts().sort_index().to_string())
    trows = []
    for T in [2018, 2019, 2020, 2021, 2022, 2023]:
        early = y_df[y_df["year"] <= T]
        late = y_df[y_df["year"] > T]
        if len(early) < 50 or len(late) < 50:
            continue
        for p in ["in_degree", "n_transitive_dependents", "pagerank_importance"]:
            res = {}
            for tag, d in [("early", early), ("late", late)]:
                # 自变量 = 当期快照的结构指标；因变量 = 该时期披露漏洞的级联广度。
                # 这不是"用历史网络预测未来"——网络只有当期一份，所以是**分组关联**：
                # 比较同一组结构指标在早/晚两个披露分组上的秩相关是否相近。
                x = d["package"].map(nm[p]).astype(float).to_numpy()
                yv = d["y"].astype(float).to_numpy()
                ok = np.isfinite(x) & np.isfinite(yv)
                rho, lo, hi = boot_spearman(x[ok], yv[ok], n=300)
                res[f"rho_{tag}"] = rho
                res[f"ci_lo_{tag}"] = lo
                res[f"ci_hi_{tag}"] = hi
                res[f"n_{tag}"] = int(ok.sum())
            res.update({"split_year": T, "predictor": p})
            trows.append(res)
    if trows:
        tdf = pd.DataFrame(trows)[["split_year", "predictor", "n_early", "rho_early",
                                   "ci_lo_early", "ci_hi_early", "n_late", "rho_late",
                                   "ci_lo_late", "ci_hi_late"]]
        tdf.to_csv(ensure_parent(TBL / "t25_temporal_holdout.csv"), index=False, encoding="utf-8")
        log("检验 2（时间外推，结构固定/事件分期）:\n" + tdf.round(4).to_string(index=False))

    # ---------------------------------------------------------- 检验 3：独立事件维度
    cnt = osv.groupby("package")["advisory_id"].nunique().rename("n_advisories")
    d = nm.join(cnt, how="left").fillna({"n_advisories": 0})
    d = d[d["in_seed"]]
    irows = []
    for p in ["in_degree", "n_transitive_dependents", "pagerank_importance", "downloads"]:
        x = d[p].astype(float).to_numpy()
        yv = d["n_advisories"].astype(float).to_numpy()
        ok = np.isfinite(x) & np.isfinite(yv)
        rho, lo, hi = boot_spearman(x[ok], yv[ok], n=500)
        irows.append({"predictor": p, "spearman_rho_vs_n_advisories": rho,
                      "ci95_low": lo, "ci95_high": hi, "n": int(ok.sum())})
    idf = pd.DataFrame(irows).sort_values("spearman_rho_vs_n_advisories", ascending=False)
    idf.to_csv(ensure_parent(TBL / "t25_incident_frequency.csv"), index=False, encoding="utf-8")
    log("检验 3（结构重要性 vs 历史 advisory 数量）:\n" + idf.round(4).to_string(index=False))

    # 双高风险：既是结构瓶颈，又在历史上多次出过 advisory
    # 注意：n_advisories 的 90% 分位数是 0（多数包从没出过 advisory），
    # 所以"历史多灾"必须用 > 0 + 分位数 的组合定义，否则会把全体都判成 True。
    q_thr = 0.90
    d2 = d.copy()
    d2["is_bottleneck"] = (d2["n_transitive_dependents"]
                           >= d2["n_transitive_dependents"].quantile(q_thr))
    pos = d2.loc[d2["n_advisories"] > 0, "n_advisories"]
    thr_inc = max(1.0, float(pos.quantile(q_thr)) if len(pos) else 1.0)
    d2["is_incident_prone"] = d2["n_advisories"] >= thr_inc
    log(f"结构瓶颈阈值 = 传递依赖者数 P90 = "
        f"{d2['n_transitive_dependents'].quantile(q_thr):.0f}；"
        f"历史多灾阈值 = advisory 数 P90（限 >0 子集）= {thr_inc:.0f}")
    tab = pd.crosstab(d2["is_bottleneck"], d2["is_incident_prone"],
                      margins=True, margins_name="合计")
    tab.to_csv(ensure_parent(TBL / "t25_dual_risk_crosstab.csv"), encoding="utf-8")
    log("双高风险交叉表（行=结构瓶颈，列=历史多灾）:\n" + tab.to_string())
    both = d2[d2["is_bottleneck"] & d2["is_incident_prone"]].sort_values(
        "n_transitive_dependents", ascending=False)
    log(f"双重高风险包 {len(both)} 个（结构瓶颈 {int(d2['is_bottleneck'].sum())} 个中的 "
        f"{len(both) / max(1, int(d2['is_bottleneck'].sum())):.2%}，"
        f"而全体种子中历史多灾比例仅 "
        f"{d2['is_incident_prone'].mean():.2%}）；Top10: {both.index[:10].tolist()}")
    both.head(50).reset_index().to_csv(
        ensure_parent(TBL / "t25_dual_risk_top50.csv"), index=False, encoding="utf-8")

    log("写出 t25_* 表")
    log.close()


if __name__ == "__main__":
    main()
