"""
Step 27 —— 敏感性：介数中心性的**采样稳定性**（评审意见 B-04）

问题
----
介数中心性（betweenness）的精确算法是 O(V·E)。本图 V=15,718、E=47,273，
朴素的"不可行"假设让我们一开始用了 k 点采样近似（k=2000）。但是：

    ★ "k 点采样"是一个**随机估计**。它带来的误差不是小事——
      它决定了"瓶颈榜前 50 名"里有多少条是噪声。
      如果一篇分析用抽样介数排出了"关键包 Top-20"却不报采样稳定性，
      那个榜单本身是不可复核的。

本脚本做三件事：

  1. 实测精确全源介数需要多久（本图约 50 秒 —— 也就是说"不可行"这个假设本身
     就是错的，我们**完全可以直接算精确值**）。
  2. 对 k ∈ {500, 1000, 2000} × seed ∈ {7, 101, 2024} 各跑一次采样介数，
     与精确值比较：Spearman、Top-10/20/50/100 的命中数、以及不同 seed 之间的
     集合重叠（Jaccard）。
  3. 给出结论口径：main pipeline 现在用**精确值**（config 里 betweenness_k=0），
     采样只作为"如果我们当初偷懒会错多少"的敏感性证据保留。

产出
----
results/tables/t27_betweenness_stability.csv   每 (k, seed) 一行：与精确值的秩相关与 Top-N 命中
results/tables/t27_betweenness_seed_overlap.csv 同一 k 下不同 seed 之间的 Top-N 重叠
results/tables/t27_betweenness_runtime.json    运行时间与计算模式记账
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (abspath, CFG, ensure_parent, Log, rank_by,
                    read_csv_safe, write_json)

log = Log("27_exp_centrality_stability")
TBL = abspath(CFG["tables"]["outdir"])
B = CFG["experiments"]["bottleneck"]
TOPS = [10, 20, 50, 100]


def load_graph() -> nx.DiGraph:
    nodes = read_csv_safe("data/processed/nodes.csv")
    edges = read_csv_safe("data/processed/edges.csv")
    G = nx.DiGraph()
    G.add_nodes_from(nodes["name"].tolist())
    G.add_edges_from(zip(edges["src"], edges["dst"]))
    return G


def top_set(scores: dict[str, float], k: int) -> list[str]:
    """确定性 Top-k（同值按名称字典序），与项目其他地方的 rank_by 口径一致。"""
    return rank_by({n: float(v) for n, v in scores.items()})[:k]


def main() -> None:
    G = load_graph()
    n, m = G.number_of_nodes(), G.number_of_edges()
    log(f"图 {n} 节点 / {m} 边")

    # ---------------------------------------------------------- 1) 精确介数
    log("计算**精确**全源介数（这本身就是对『精确不可行』假设的检验）…")
    t0 = time.time()
    exact = nx.betweenness_centrality(G, k=None, normalized=True)
    t_exact = time.time() - t0
    log(f"  精确介数用时 {t_exact:.1f}s")
    ex_top = {k: set(top_set(exact, k)) for k in TOPS}
    ex_vec = np.array([exact[v] for v in G.nodes()], dtype=float)

    rows = [{
        "k": int(n), "seed": -1, "mode": "exact",
        "runtime_s": t_exact,
        "spearman_vs_exact": 1.0, "pearson_vs_exact": 1.0,
        **{f"top{k}_hits": k for k in TOPS},
        **{f"top{k}_jaccard": 1.0 for k in TOPS},
    }]

    # ---------------------------------------------------------- 2) 采样介数
    cache: dict[tuple[int, int], dict[str, float]] = {}
    for k in B["betweenness_stability_k"]:
        for sd in B["betweenness_stability_seeds"]:
            t0 = time.time()
            b = nx.betweenness_centrality(G, k=int(k), normalized=True, seed=int(sd))
            dt = time.time() - t0
            cache[(int(k), int(sd))] = b
            bv = np.array([b[v] for v in G.nodes()], dtype=float)
            rho = stats.spearmanr(ex_vec, bv).statistic
            pear = stats.pearsonr(ex_vec, bv).statistic
            rec = {"k": int(k), "seed": int(sd), "mode": "sampled", "runtime_s": dt,
                   "spearman_vs_exact": float(rho), "pearson_vs_exact": float(pear)}
            for t in TOPS:
                s = set(top_set(b, t))
                rec[f"top{t}_hits"] = len(s & ex_top[t])
                rec[f"top{t}_jaccard"] = len(s & ex_top[t]) / len(s | ex_top[t])
            rows.append(rec)
            log(f"  k={k} seed={sd}: ρ={rho:.4f}  用时 {dt:.1f}s  "
                f"Top50 命中 {rec['top50_hits']}/50")

    sdf = pd.DataFrame(rows)
    sdf.to_csv(ensure_parent(TBL / "t27_betweenness_stability.csv"),
               index=False, encoding="utf-8")
    log("介数采样 vs 精确:\n" + sdf.round(4).to_string(index=False))

    # ---------------------------------------------------------- 3) seed 间重叠
    orows = []
    for k in B["betweenness_stability_k"]:
        seeds = [int(s) for s in B["betweenness_stability_seeds"]]
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                a = cache[(int(k), seeds[i])]
                b = cache[(int(k), seeds[j])]
                rec = {"k": int(k), "seed_a": seeds[i], "seed_b": seeds[j]}
                for t in TOPS:
                    sa, sb = set(top_set(a, t)), set(top_set(b, t))
                    rec[f"top{t}_jaccard"] = len(sa & sb) / len(sa | sb)
                rec["spearman_pair"] = float(stats.spearmanr(
                    [a[v] for v in G.nodes()], [b[v] for v in G.nodes()]).statistic)
                orows.append(rec)
    odf = pd.DataFrame(orows)
    odf.to_csv(ensure_parent(TBL / "t27_betweenness_seed_overlap.csv"),
               index=False, encoding="utf-8")
    log("同 k 不同 seed 的 Top-N 重叠:\n" + odf.round(4).to_string(index=False))

    write_json(ensure_parent(TBL / "t27_betweenness_runtime.json"), {
        "n_nodes": n, "n_edges": m,
        "exact_runtime_s": t_exact,
        "sampled_runtime_s": {f"k={k},seed={s}": v["runtime_s"] for (k, s), v in
                              zip(cache.keys(), [{"runtime_s": r["runtime_s"]}
                                                  for r in rows if r["mode"] == "sampled"])},
        "main_pipeline_mode": "exact" if not B.get("betweenness_k") else
                              f"sampled k={B['betweenness_k']}",
        "conclusion_rule": "若 top50_hits 明显低于 50 或 seed 间 Top-50 Jaccard 明显低于 1，"
                           "则抽样介数不足以支撑榜单，主流程必须用精确值。",
    })
    log("写出 t27_* 表")
    log.close()


if __name__ == "__main__":
    main()
