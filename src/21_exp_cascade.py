"""
Step 21 —— 实验二：把"级联影响面"与"瓶颈分数"一次算出来

本脚本产出 data/processed/node_metrics.parquet，一行为一个包：

│ 列名                        │ 定义                                                     │
├─────────────────────────────┼──────────────────────────────────────────────────────────┤
│ in_degree                   │ 直接依赖它的包数（入度）                                 │
│ n_transitive_dependents ★   │ **传递**依赖它的种子包数（已解析闭包口径，精确）         │
│ n_transitive_all_nodes      │ 传递依赖它的全部节点数（直接边图口径，含榜外叶子）       │
│ cascade_depth ★             │ 它出问题后，级联沿依赖链向上最多爬几跳（反向最长路）     │
│ pagerank_importance ★       │ 依赖流图 G 上的 PageRank = "被多少重要包依赖"            │
│ pagerank_inherited          │ 反向图 G^R 上的 PageRank = "它依赖了多少重要包"（风险继承）│
│ betweenness                 │ 有向介数中心性（k 点采样近似，见配置）                   │
│ kcore                       │ 无向投影上的 k-core（核心层深度）                        │
│ weighted_exposure ★         │ 传递依赖者的下载量之和（真实安装量暴露）                 │
│ reach_frac                  │ n_transitive_dependents / 种子总数                        │

★ = 本项目作为"瓶颈分数"的主候选。

★★ 关于 PageRank 方向的说明（很容易搞反，所以写在这里）★★
* 我们的图 G 里，边是 `依赖者 -> 依赖`（dependency flow）。
* 级联的**传播**方向与之相反（被依赖者出问题 -> 依赖者受累）。
* 想度量"一个包作为底层被依赖者的枢纽程度"，正确做法是在 **G 本身**上跑 PageRank：
      rank(v) = Σ_{u->v} rank(u)/outdeg(u)
  即"被越多重要的包依赖 -> 分数越高"。这就是本文所说的 bottleneck PageRank。
  （有些文献把它叫 "reverse PageRank"，指的是"相对于传播方向取反"，同一个量。）
* 反过来，在 G^R 上跑 PageRank 度量的是"v 依赖的那些包有多重要"，
  那是**风险继承（fragility）**，不是瓶颈。两者含义相反，因此分开命名、分开报告。

口径说明
--------
* `n_transitive_dependents` 用 deps.dev 的已解析闭包直接统计，
  不依赖直接边图是否完整，因此没有"图截断"偏差。
* `n_transitive_all_nodes` 用直接边图的反向可达集统计；
  对于榜外叶子节点（本项目的 725 个），我们没有它们的直接依赖信息，
  通过它们的更深路径会漏掉。两者之差被单独记账，不假装没有。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (abspath, bits_to_mask, CFG, ensure_parent, Log,
                    read_csv_safe, read_universe, write_json)

log = Log("21_exp_cascade")
TBL = abspath(CFG["tables"]["outdir"])
E = CFG["experiments"]


# ------------------------------------------------------------------ 数据加载
def load_graph() -> nx.DiGraph:
    nodes = read_csv_safe("data/processed/nodes.csv", keep_default_na=False,
                        na_values=[""])
    edges = read_csv_safe("data/processed/edges.csv")
    G = nx.DiGraph()
    G.add_nodes_from(nodes["name"].tolist())
    G.add_edges_from(zip(edges["src"], edges["dst"]))
    uni = read_universe()
    seed_set = set(uni["project"])
    nx.set_node_attributes(G, {v: (v in seed_set) for v in G.nodes()}, "in_seed")
    # 下载量权重：榜外包无数据，显式置 None，下游按 0 处理（不臆造）
    dl = {}
    for r in nodes.itertuples(index=False):
        v = r.downloads
        dl[r.name] = None if (v is None or (isinstance(v, float) and np.isnan(v)) or v == "") else float(v)
    nx.set_node_attributes(G, dl, "downloads")
    return G


def closure_based_breadth() -> pd.DataFrame:
    cl = pd.read_parquet(abspath("data/processed/closures.parquet"))
    cl = cl[cl["relation"] != "SELF"]
    seeds = set(read_universe()["project"])
    cl = cl[cl["seed"].isin(seeds)]
    return (cl.groupby("dep", observed=True)
              .agg(n_transitive_dependents=("seed", "nunique"),
                   n_closure_records=("seed", "size"))
              .reset_index().rename(columns={"dep": "name"}))


# ------------------------------------------------------------------ 反向可达（位集）
def reverse_reach(G: nx.DiGraph):
    """返回 (反向可达集大小, 反向可达集下载量之和, 位集矩阵, 节点顺序)。

    依赖图在理论上应是 DAG，但实测有 20 个小强连通分量（条件依赖互相引用）。
    统一做法：先凝缩 SCC（环内节点必然一起触发级联），在凝缩 DAG 上
    按**前向拓扑序**把"依赖者的可达集"向上合并。对 DAG 而言凝缩是恒等映射，
    所以这套代码对两种情况都正确。
    """
    names = list(G.nodes())
    idx = {v: i for i, v in enumerate(names)}
    n = len(names)
    if n == 0:
        return {}, {}, None, []
    nw = (n + 63) // 64

    dl = np.array([G.nodes[v].get("downloads") or 0.0 for v in names], dtype=np.float64)
    dl = np.nan_to_num(dl, nan=0.0)

    C = nx.condensation(G)
    cbits: dict[int, np.ndarray] = {}
    for c in nx.topological_sort(C):
        b = np.zeros(nw, dtype=np.uint64)
        for m in C.nodes[c]["members"]:
            b[idx[m] >> 6] |= np.uint64(1) << np.uint64(idx[m] & 63)
        # C.predecessors(c) = 那些"依赖 c 中节点"的分量 = 依赖者
        for q in C.predecessors(c):
            b |= cbits[q]
        cbits[c] = b

    bits = np.zeros((n, nw), dtype=np.uint64)
    for c, d in C.nodes(data=True):
        for m in d["members"]:
            bits[idx[m]] = cbits[c]

    cnt: dict[str, int] = {}
    wsum: dict[str, float] = {}
    for v in names:
        mask = bits_to_mask(bits[idx[v]], n)
        cnt[v] = int(mask.sum()) - 1
        wsum[v] = float(dl[mask].sum()) - float(dl[idx[v]])
    return cnt, wsum, bits, names


# ------------------------------------------------------------------ 级联深度
def cascade_depth(G: nx.DiGraph) -> dict[str, int]:
    """depth(v) = max over 依赖者 u of v: 1 + depth(u)；即"向上最多爬几跳"。

    需要先算出依赖者的深度，而依赖者在拓扑序里更靠前 => **前向拓扑序**。
    （这一行顺序写错就会让所有深度变成 0 或 1——本项目在自查时踩过这个坑。）
    """
    C = nx.condensation(G)
    dc: dict[int, int] = {}
    for c in nx.topological_sort(C):
        best = 0
        for q in C.predecessors(c):        # q 依赖 c 中的节点
            best = max(best, 1 + dc.get(q, 0))
        dc[c] = best
    out: dict[str, int] = {}
    for c, d in C.nodes(data=True):
        for m in d["members"]:
            out[m] = dc[c]
    return out


# ------------------------------------------------------------------ 主流程
def main() -> None:
    G = load_graph()
    n = G.number_of_nodes()
    uni = read_universe()
    seeds = set(uni["project"])
    log(f"节点 {n}，边 {G.number_of_edges()}，种子 {len(seeds)}")

    # 1) 精确广度（闭包口径）
    breadth = closure_based_breadth()
    log(f"闭包口径：{len(breadth)} 个包至少被 1 个种子传递依赖")

    # 2) 级联深度
    depth = cascade_depth(G)
    dv = np.array(list(depth.values()))
    log(f"级联深度：max={dv.max()}  p99={np.percentile(dv,99):.0f}  mean={dv.mean():.2f}")

    # 3) 图口径反向可达 + 加权暴露
    log("计算反向可达集（SCC 凝缩 + 位集向上合并）…")
    reach_cnt, reach_w, bits, rnames = reverse_reach(G)
    np.save(ensure_parent(abspath("data/processed/reach_bits.npy")), bits)
    with open(ensure_parent(abspath("data/processed/reach_nodes.json")), "w",
              encoding="utf-8") as f:
        json.dump(rnames, f)
    log(f"  位集 {bits.shape} 已落盘（{bits.nbytes/1e6:.1f} MB）")

    # 4) PageRank（两个方向，含义相反，分开命名）
    log("计算 PageRank（依赖流方向 = 瓶颈重要性）…")
    pr_imp = nx.pagerank(G, alpha=E["bottleneck"]["reverse_pagerank_alpha"],
                         max_iter=300, tol=1e-12)
    log("计算 PageRank（反向 = 风险继承）…")
    pr_inh = nx.pagerank(G.reverse(copy=True), alpha=E["bottleneck"]["reverse_pagerank_alpha"],
                         max_iter=300, tol=1e-12)

    # 5) 介数（k 点采样近似）
    k = E["bottleneck"]["betweenness_k"]
    kk = None if (k is None or k <= 0 or k >= n) else int(k)
    log(f"计算介数中心性（k={kk} 采样近似）…")
    btw = nx.betweenness_centrality(G, k=kk, normalized=True,
                                    seed=E["bottleneck"]["betweenness_seed"])

    # 6) k-core
    log("计算 k-core…")
    core = nx.core_number(G.to_undirected(as_view=False))

    # 7) 合并
    deg = pd.read_parquet(abspath("data/processed/degrees.parquet"))
    df = deg.merge(breadth, on="name", how="left")
    df["n_transitive_dependents"] = df["n_transitive_dependents"].fillna(0).astype(int)
    df["n_closure_records"] = df["n_closure_records"].fillna(0).astype(int)
    df["cascade_depth"] = df["name"].map(depth).fillna(0).astype(int)
    df["n_transitive_all_nodes"] = df["name"].map(reach_cnt).fillna(0).astype(int)
    df["weighted_exposure"] = df["name"].map(reach_w).fillna(0.0)
    df["pagerank_importance"] = df["name"].map(pr_imp).fillna(0.0)
    df["pagerank_inherited"] = df["name"].map(pr_inh).fillna(0.0)
    df["betweenness"] = df["name"].map(btw).fillna(0.0)
    df["kcore"] = df["name"].map(core).fillna(0).astype(int)
    n_seed = max(1, int(df["in_seed"].sum()))
    df["reach_frac"] = df["n_transitive_dependents"] / n_seed

    out = ensure_parent(abspath("data/processed/node_metrics.parquet"))
    df.to_parquet(out, index=False)
    log(f"-> {out} ({len(df)} 行)")

    # 8) 口径差异记账
    both = df[(df["n_transitive_dependents"] > 0) & (df["n_transitive_all_nodes"] > 0)]
    gap = both["n_transitive_all_nodes"] - both["n_transitive_dependents"]
    trunc = {
        "n_nodes": int(len(df)),
        "n_with_seed_dependents": int((df["n_transitive_dependents"] > 0).sum()),
        "n_with_graph_dependents": int((df["n_transitive_all_nodes"] > 0).sum()),
        "n_comparable": int(len(both)),
        "median_graph_minus_closure": float(gap.median()) if len(both) else 0.0,
        "mean_graph_minus_closure": float(gap.mean()) if len(both) else 0.0,
        "p95_graph_minus_closure": float(np.percentile(gap, 95)) if len(both) else 0.0,
        "spearman_graph_vs_closure": float(both["n_transitive_all_nodes"].corr(
            both["n_transitive_dependents"], method="spearman")) if len(both) else None,
    }
    log("口径差异（图口径 − 闭包口径）：" + json.dumps(trunc, ensure_ascii=False))

    # 9) 指标一致性
    metrics = ["in_degree", "n_transitive_dependents", "cascade_depth",
               "pagerank_importance", "pagerank_inherited", "betweenness",
               "kcore", "weighted_exposure"]
    rho = df[metrics].astype(float).corr(method="spearman")
    ensure_parent(TBL / "t21_metric_spearman.csv")
    rho.to_csv(TBL / "t21_metric_spearman.csv", encoding="utf-8")
    log("指标间 Spearman 秩相关:\n" + rho.round(3).to_string())

    # 10) 各种口径的瓶颈榜
    cols = ["name", "in_seed", "downloads", "in_degree", "n_transitive_dependents",
            "cascade_depth", "pagerank_importance", "pagerank_inherited",
            "betweenness", "kcore", "weighted_exposure", "reach_frac"]
    for key in ["n_transitive_dependents", "pagerank_importance", "betweenness",
                "weighted_exposure", "cascade_depth", "in_degree"]:
        df.sort_values(key, ascending=False).head(30)[cols].to_csv(
            ensure_parent(TBL / f"t21_top30_by_{key}.csv"), index=False, encoding="utf-8")

    write_json(abspath("results/tables/t21_truncation.json"), trunc)
    log("写出 t21_* 表")
    log.close()


if __name__ == "__main__":
    main()
