"""
Step 30 —— 生成所有图表（一次跑完，供报告与 PPT 直接引用）。

约定：
  * 每张图都内嵌"这是什么口径"的副标题，避免脱离上下文被误读；
  * 所有涉及随机性的图都画出波动带（±1 s.d. 或 bootstrap CI）；
  * 中文用 Microsoft YaHei / SimHei 回退，缺字时可用英文标签。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import abspath, CFG, ensure_parent, Log, read_csv_safe

log = Log("30_figures")
FIG = abspath(CFG["figures"]["outdir"])
TBL = abspath(CFG["tables"]["outdir"])
DPI = CFG["figures"]["dpi"]

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 110,
    "savefig.dpi": DPI,
    "axes.grid": True,
    "grid.alpha": .3,
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "figure.autolayout": False,
})
# 统一配色
C = {"and": "#c0392b", "or": "#2980b9", "random": "#7f8c8d", "obs": "#111111",
     "null": "#e67e22", "s": "#8e44ad", "v": "#16a085", "warn": "#d35400"}


def save(fig, name):
    p = ensure_parent(FIG / f"{name}.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"  -> {name}.png")


def ccdf(a):
    a = np.sort(np.asarray(a, dtype=float))
    n = len(a)
    return a, 1.0 - np.arange(n) / n


# ------------------------------------------------------------------ F1 度分布
def f1_degree():
    deg = pd.read_parquet(abspath("data/processed/degrees.parquet"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, lab, col_c in [
        (axes[0], "in_degree", "入度（被多少包直接依赖）", C["and"]),
        (axes[1], "out_degree", "出度（直接依赖多少包）", C["or"]),
    ]:
        x, y = ccdf(deg.loc[deg[col] > 0, col])
        ax.loglog(x, y, "o", ms=3, alpha=.55, color=col_c)
        ax.set_xlabel(lab)
        ax.set_ylabel("P(X ≥ x)")
        ax.set_title(f"{lab} — 互补累积分布")
    fig.suptitle("图 1  PyPI 依赖网络的度分布：入度重尾（少数包被极多包依赖），出度窄（多数包只依赖个位数包）",
                 fontsize=11.5, y=1.02)
    save(fig, "f01_degree_ccdf")


# ------------------------------------------------------------------ F2 级联影响面
def f2_breadth():
    nm = pd.read_parquet(abspath("data/processed/node_metrics.parquet"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    x, y = ccdf(nm.loc[nm["n_transitive_dependents"] > 0, "n_transitive_dependents"])
    ax.loglog(x, y, "o", ms=3, alpha=.6, color=C["and"])
    ax.set_xlabel("传递依赖它的种子包数 |R(v)|")
    ax.set_ylabel("P(X ≥ x)")
    ax.set_title("级联影响面（广度）：重尾")
    ax = axes[1]
    v = np.sort(nm["n_transitive_dependents"].to_numpy())[::-1]
    cum = np.cumsum(v) / v.sum()
    ax.plot(np.arange(1, len(v) + 1) / len(v), cum, color=C["s"], lw=2)
    for frac in [0.01, 0.05, 0.10]:
        k = max(1, int(len(v) * frac))
        ax.axvline(frac, ls=":", color="gray", lw=1)
        ax.annotate(f"前 {frac:.0%} 的包\n承担 {cum[k-1]:.0%} 的暴露",
                    xy=(frac, cum[k - 1]), xytext=(frac + .08, cum[k - 1] - .18),
                    fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("包的累计比例（按影响面从大到小排序）")
    ax.set_ylabel("累计暴露占比")
    ax.set_title("暴露的高度集中性（Lorenz 型曲线）")
    fig.suptitle("图 2  “一个底层包出问题会波及多少包”——广度呈重尾，且暴露极度集中",
                 fontsize=11.5, y=1.02)
    save(fig, "f02_cascade_breadth")


# ------------------------------------------------------------------ F3 级联深度
def f3_depth():
    nm = pd.read_parquet(abspath("data/processed/node_metrics.parquet"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    ax = axes[0]
    d = nm["cascade_depth"]
    vc = d.value_counts().sort_index()
    ax.bar(vc.index, vc.values, color=C["or"], alpha=.8)
    ax.set_xlabel("级联向上传播的最大跳数（深度）")
    ax.set_ylabel("包数")
    ax.set_title("级联深度的分布")
    ax = axes[1]
    sub = nm[nm["cascade_depth"] > 0]
    ax.scatter(sub["cascade_depth"], sub["n_transitive_dependents"], s=4, alpha=.25,
               color=C["s"], rasterized=True)
    ax.set_yscale("log")
    ax.set_xlabel("深度（跳数）")
    ax.set_ylabel("广度（受影响包数）")
    ax.set_title("深 vs 广：二者不共变")
    fig.suptitle("图 3  级联的“深度”与“广度”是两个独立维度——深而窄、广而浅都很常见",
                 fontsize=11.5, y=1.02)
    save(fig, "f03_cascade_depth")


# ------------------------------------------------------------------ F4 指标一致性
def f4_metric_corr():
    rho = pd.read_csv(TBL / "t21_metric_spearman.csv", index_col=0)
    ren = {"in_degree": "入度", "n_transitive_dependents": "传递依赖者数",
           "cascade_depth": "级联深度", "pagerank_importance": "PageRank(重要性)",
           "pagerank_inherited": "PageRank(风险继承)",
           "betweenness": "介数", "kcore": "k-core", "weighted_exposure": "加权暴露量"}
    rho = rho.rename(index=ren, columns=ren)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(rho.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(rho)), rho.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(rho)), rho.index)
    for i in range(len(rho)):
        for j in range(len(rho)):
            ax.text(j, i, f"{rho.iat[i,j]:.2f}", ha="center", va="center",
                    fontsize=8.5, color="black")
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("图 4  “哪个包重要”——不同指标给出的排名并不一致\n"
                 "（ρ=1 表示排名完全一致；灰色带表示两组结论不可互换）", fontsize=11)
    save(fig, "f04_metric_spearman")


# ------------------------------------------------------------------ F5 结构 vs 版本
def f5_struct_vs_version():
    imp = read_csv_safe(TBL / "t22_advisory_impact.csv")
    d = imp[imp["a_struct"] > 0].copy()
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(1, 3, 1)
    jitter = lambda a: a + np.random.default_rng(0).uniform(-.15, .15, size=len(a))
    ax.scatter(jitter(d["a_struct"].to_numpy(float)), d["a_version"], s=7, alpha=.25,
               color=C["s"], rasterized=True)
    lim = [0.8, max(d["a_struct"].max(), 2) * 1.3]
    ax.plot(lim, lim, "--", color="gray", lw=1, label="无高估（y=x）")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(0.8, lim[1])
    ax.set_xlabel("结构口径受影响包数（上界）")
    ax.set_ylabel("版本约束感知受影响包数（可实现）")
    ax.set_title("逐条 advisory")
    ax.legend()

    ax = fig.add_subplot(1, 3, 2)
    r = d[d["a_version"] > 0].copy()
    r["ratio"] = r["a_struct"] / r["a_version"]
    ax.hist(np.log10(r["ratio"].clip(lower=1)), bins=40, color=C["v"], alpha=.85)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("log10(结构口径 / 版本口径)")
    ax.set_ylabel("advisory 数")
    ax.set_title(f"过近似倍数（中位数 {r['ratio'].median():.2f}×）")

    ax = fig.add_subplot(1, 3, 3)
    zero = (d["a_version"] == 0).mean()
    ax.bar(["实际受影响", "结构上“命中”但\n实际不受影响"],
           [1 - zero, zero], color=[C["v"], C["warn"]], alpha=.9)
    for i, v in enumerate([1 - zero, zero]):
        ax.text(i, v + .02, f"{v:.1%}", ha="center", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("占比")
    ax.set_title("有多少 advisory 是“空惊一场”")
    fig.suptitle("图 5  把依赖扫描的“结构上界”收紧成“版本约束下的可实现影响”——高估是系统性且巨大的",
                 fontsize=11.5, y=1.03)
    save(fig, "f05_struct_vs_version")


# ------------------------------------------------------------------ F6 机制分解
def f6_mechanism():
    p = TBL / "t22_mechanism.csv"
    if not p.exists():
        log("  跳过 f06（无 t22_mechanism.csv）")
        return
    m = read_csv_safe(p).dropna(subset=["frac_already_patched"])
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    ax.hist(m["frac_already_patched"], bins=30, color=C["v"], alpha=.85)
    ax.axvline(m["frac_already_patched"].median(), color=C["and"], lw=2,
               label=f"中位数 {m['frac_already_patched'].median():.2f}")
    ax.set_xlabel("依赖者中「解析到的版本已在修复版本之后」的比例")
    ax.set_ylabel("advisory 数")
    ax.legend()
    ax.set_title("图 6  “高估”的机制：上游一升级，下游的传递依赖就自动落到修复版本之后\n"
                 "（因此它们从未真实暴露，却在结构扫描里被记为受影响）", fontsize=11)
    save(fig, "f06_mechanism_patched")


# ------------------------------------------------------------------ F7 S(q) 曲线
def f7_resilience():
    c = read_csv_safe(TBL / "t23_curves_summary.csv")
    order = ["by_transitive_dependents", "by_reverse_pagerank", "by_in_degree",
             "by_betweenness", "by_downloads", "by_kcore", "random"]
    name = {"by_transitive_dependents": "按传递依赖者", "by_pagerank_importance": "按 PageRank(重要性)",
            "by_in_degree": "按入度", "by_betweenness": "按介数",
            "by_downloads": "按下载量", "by_kcore": "按 k-core", "random": "随机"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, rule, col, title in [
        (axes[0], "S_and_mean", C["and"], "规则 A：AND（任一依赖坏 → 包坏）\n= 软件的真实语义"),
        (axes[1], "S_or_mean", C["or"], "规则 B：OR（全部依赖坏 → 包坏）\n= 食物网次级灭绝的语义"),
    ]:
        for s in order:
            sub = c[c["strategy"] == s].sort_values("q")
            if not len(sub):
                continue
            ls = "--" if s == "random" else "-"
            lw = 2.4 if s in ("by_transitive_dependents", "random") else 1.5
            ax.plot(sub["q"], sub[rule], marker="o", ms=3, ls=ls, lw=lw,
                    label=name.get(s, s), color=_palette(s))
            sd = sub[rule.replace("_mean", "_std")]
            if s == "random":
                ax.fill_between(sub["q"], sub[rule] - sd, sub[rule] + sd, color="gray", alpha=.18)
        ax.set_xlabel("移除比例 q")
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("S(q) = 仍完全可用的种子包比例")
    axes[0].legend(loc="lower left", ncol=1, fontsize=8.5)
    fig.suptitle("图 7  同一次移除，两种失效规则给出完全不同的韧性结论——"
                 "把食物网规则直接搬到软件上会严重低估脆弱性", fontsize=11.5, y=1.03)
    save(fig, "f07_resilience_rules")


def _palette(s):
    m = {"by_transitive_dependents": "#c0392b", "by_pagerank_importance": "#8e44ad",
         "by_in_degree": "#e67e22", "by_betweenness": "#16a085",
         "by_downloads": "#2c3e50", "by_kcore": "#7f8c8d", "random": "#95a5a6"}
    return m.get(s, "#333")


# ------------------------------------------------------------------ F8 零模型对照
def f8_null():
    p = TBL / "t24_null_compare.csv"
    if not p.exists():
        log("  跳过 f08（无 t24_null_compare.csv）")
        return
    n = read_csv_safe(p)
    c = read_csv_safe(TBL / "t24_null_curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    met = ["true_max", "null_max", "true_p99", "null_p99"]
    lab = ["真实\nmax", "零模型\nmax", "真实\np99", "零模型\np99"]
    cols = [C["obs"], C["null"], C["obs"], C["null"]]
    ax.bar(lab, [n[m].mean() for m in met], yerr=[n[m].std() for m in met],
           color=cols, alpha=.85, capsize=3)
    ax.set_ylabel("反向可达集规模")
    ax.set_title("级联规模：真实 vs 保度序列零模型")
    ax = axes[1]
    for mode, ls in [("random", "--"), ("by_in_degree", "-")]:
        sub = c[c["mode"] == mode].sort_values("q")
        ax.plot(sub["q"], sub["S_true"], ls=ls, lw=2.2, color=C["obs"],
                label=f"真实 · {mode}")
        ax.plot(sub["q"], sub["S_null_mean"], ls=ls, lw=1.6, color=C["null"],
                label=f"零模型 · {mode}")
        ax.fill_between(sub["q"], sub["S_null_mean"] - sub["S_null_std"],
                        sub["S_null_mean"] + sub["S_null_std"], color=C["null"], alpha=.18)
    ax.set_xlabel("移除比例 q"); ax.set_ylabel("S(q)")
    ax.legend(fontsize=8)
    ax.set_title("S(q)：真实 vs 零模型")
    ax = axes[2]
    ax.scatter(n["true_gini"], n["null_gini"], s=40, color=C["s"])
    lim = [min(n["true_gini"].min(), n["null_gini"].min()) * .95,
           max(n["true_gini"].max(), n["null_gini"].max()) * 1.02]
    ax.plot(lim, lim, "--", color="gray", lw=1)
    ax.set_xlabel("真实网络 Gini(反向可达集)")
    ax.set_ylabel("零模型 Gini")
    ax.set_title("集中度对比")
    fig.suptitle("图 8  对照实验：把“谁连谁”随机化但保持度序列后，级联结构还剩多少？",
                 fontsize=11.5, y=1.03)
    save(fig, "f08_null_model")


# ------------------------------------------------------------------ F9 预测力
def f9_predictive():
    p = TBL / "t25_predictive_power.csv"
    if not p.exists():
        log("  跳过 f09")
        return
    d = read_csv_safe(p).sort_values("spearman_rho")
    lab = {"in_degree": "入度（最便宜）", "n_transitive_dependents": "传递依赖者数",
           "pagerank_importance": "PageRank(重要性)", "betweenness": "介数",
           "kcore": "k-core", "downloads": "下载量", "cascade_depth": "级联深度"}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(d))
    ax.barh(y, d["spearman_rho"], xerr=[d["spearman_rho"] - d["ci95_low"],
                                        d["ci95_high"] - d["spearman_rho"]],
            color=[C["and"] if r > .5 else C["or"] for r in d["spearman_rho"]],
            alpha=.85, capsize=3)
    ax.set_yticks(y, [lab.get(x, x) for x in d["predictor"]])
    ax.axvline(0, color="gray", lw=1)
    ax.set_xlabel("Spearman ρ（静态指标 vs 版本感知的真实受影响包数）")
    ax.set_title("图 9  便宜的拓扑指标能否预测“真实”级联影响面？\n"
                 "（误差条为 bootstrap 95% CI）", fontsize=11)
    save(fig, "f09_predictive_power")


# ------------------------------------------------------------------ F10 Patch vs Remove
def f10_patch_remove():
    p = TBL / "t23_patch_vs_remove.csv"
    if not p.exists():
        log("  跳过 f10")
        return
    d = read_csv_safe(p)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    ax.plot(d["k"], d["exposure_patch_drop_pct"], "o-", color=C["v"], lw=2, label="情景 P：修复 k 个瓶颈包")
    ax.plot(d["k"], d["exposure_remove_drop_pct"], "s--", color=C["and"], lw=2, label="情景 R：移除 k 个瓶颈包")
    ax.set_xlabel("处理的关键包个数 k")
    ax.set_ylabel("生态总暴露量下降 %")
    ax.set_title("暴露量视角")
    ax.legend()
    ax = axes[1]
    ax.plot(d["k"], d["frac_seeds_broken"] * 100, "s-", color=C["warn"], lw=2)
    ax.set_xlabel("移除的关键包个数 k")
    ax.set_ylabel("种子包中彻底不可构建的比例 %")
    ax.set_title("“移除”的代价：可用性崩塌")
    fig.suptitle("图 10  “修复”与“移除”是两种性质不同的干预：前者降低暴露，后者摧毁可用性",
                 fontsize=11.5, y=1.03)
    save(fig, "f10_patch_vs_remove")


# ------------------------------------------------------------------ F11 双风险
def f11_dual_risk():
    p = TBL / "t25_dual_risk_top50.csv"
    if not p.exists():
        log("  跳过 f11")
        return
    nm = pd.read_parquet(abspath("data/processed/node_metrics.parquet")).set_index("name")
    osv = read_csv_safe("data/processed/osv_advisories.csv")
    cnt = osv.groupby("package")["advisory_id"].nunique()
    d = nm.join(cnt.rename("n_adv"), how="left").fillna({"n_adv": 0})
    d = d[d["in_seed"] & (d["n_transitive_dependents"] > 0)]
    fig, ax = plt.subplots(figsize=(7.6, 5))
    sc = ax.scatter(d["n_transitive_dependents"], d["n_adv"] + 1, s=8, alpha=.3,
                    c=np.log10(d["downloads"].fillna(1) + 1), cmap="viridis", rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("结构瓶颈分数：传递依赖者数")
    ax.set_ylabel("历史 advisory 数 + 1")
    top = d.sort_values("n_transitive_dependents", ascending=False).head(8)
    for nmx, r in top.iterrows():
        ax.annotate(nmx, (r["n_transitive_dependents"], r["n_adv"] + 1),
                    fontsize=7.5, alpha=.9)
    fig.colorbar(sc, ax=ax, label="log10(下载量)")
    ax.set_title("图 11  “结构上关键”与“历史上多灾”是否重合？\n"
                 "（右上象限 = 双重高风险，应优先治理）", fontsize=11)
    save(fig, "f11_dual_risk")


# ------------------------------------------------------------------ F12 局部网络
def f12_subgraph():
    nodes = read_csv_safe("data/processed/nodes.csv")
    edges = read_csv_safe("data/processed/edges.csv")
    nm = pd.read_parquet(abspath("data/processed/node_metrics.parquet"))
    import networkx as nx

    top = nm.sort_values("n_transitive_dependents", ascending=False).head(45)["name"].tolist()
    keep = set(top)
    e = edges[edges["src"].isin(keep) & edges["dst"].isin(keep)]
    G = nx.DiGraph()
    G.add_nodes_from(keep)
    G.add_edges_from(zip(e["src"], e["dst"]))
    if G.number_of_edges() < 20:
        log("  跳过 f12（子图过稀）")
        return
    sz = nm.set_index("name")["n_transitive_dependents"]
    pos = nx.spring_layout(G, seed=7, k=.6)
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ns = [40 + 260 * np.log1p(sz.get(v, 1)) / np.log1p(sz.max()) for v in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=7, width=.7,
                           edge_color="#888", alpha=.5, connectionstyle="arc3,rad=0.08")
    nc = [sz.get(v, 1) for v in G.nodes()]
    sc = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=ns, node_color=nc,
                                cmap="YlOrRd", linewidths=.4, edgecolors="black")
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={v: v for v in G.nodes() if sz.get(v, 0) > 0},
                            font_size=6.5)
    fig.colorbar(sc, ax=ax, label="传递依赖者数")
    ax.set_axis_off()
    ax.set_title("图 12  Top-45 瓶颈包之间的依赖骨架（箭头 A→B 表示 A 依赖 B）\n"
                 "越红越大 = 一旦出问题波及越广", fontsize=11)
    save(fig, "f12_bottleneck_subgraph")


if __name__ == "__main__":
    for fn in [f1_degree, f2_breadth, f3_depth, f4_metric_corr, f5_struct_vs_version,
               f6_mechanism, f7_resilience, f8_null, f9_predictive, f10_patch_remove,
               f11_dual_risk, f12_subgraph]:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            log(f"  [失败] {fn.__name__}: {e}")
    log("图生成完成")
    log.close()
