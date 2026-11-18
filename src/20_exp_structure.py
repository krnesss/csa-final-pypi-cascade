"""
Step 20 —— 实验一：网络结构描述与"依赖深度/广度"基线。

回答：这个生态在结构上到底是什么样子？哪些统计量是后面实验必须控制的混杂？
产出：
  results/tables/t20_structure.csv        结构指标汇总
  results/tables/t20_degree_stats.csv      入度/出度分布分位数
  results/tables/t20_depth.csv             每个种子的依赖深度与闭包规模
  results/tables/t20_hubs.csv              Top-40 枢纽包（按被依赖数）
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import abspath, CFG, ensure_parent, Log, read_csv_safe

log = Log("20_exp_structure")
TBL = abspath(CFG["tables"]["outdir"])


def load_graph() -> nx.DiGraph:
    nodes = read_csv_safe("data/processed/nodes.csv")
    edges = read_csv_safe("data/processed/edges.csv")
    G = nx.DiGraph()
    for r in nodes.itertuples(index=False):
        G.add_node(r.name, in_seed=bool(r.in_seed),
                   downloads=None if pd.isna(r.downloads) else float(r.downloads),
                   rank=None if pd.isna(r.rank) else int(r.rank))
    for r in edges.itertuples(index=False):
        G.add_edge(r.src, r.dst, rel=r.rel)
    return G


def ccdf(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.sort(np.asarray(vals))
    n = len(v)
    return v, 1.0 - np.arange(n) / n


def main() -> None:
    G = load_graph()
    n, e = G.number_of_nodes(), G.number_of_edges()
    seeds = [v for v, d in G.nodes(data=True) if d.get("in_seed")]
    log(f"节点 {n}, 边 {e}, 种子 {len(seeds)}")

    out_deg = dict(G.out_degree())          # 依赖了多少包
    in_deg = dict(G.in_degree())            # 被多少包直接依赖
    n_direct_dependents = in_deg

    # ---------------- 度分布
    def qstats(arr, name):
        a = np.array(arr, dtype=float)
        return {
            "metric": name,
            "mean": a.mean(), "std": a.std(),
            "p50": np.percentile(a, 50), "p90": np.percentile(a, 90),
            "p99": np.percentile(a, 99), "max": a.max(),
            "frac_zero": float((a == 0).mean()),
            "gini": _gini(a),
        }

    rows = [
        qstats(list(out_deg.values()), "out_degree (依赖数)"),
        qstats(list(in_deg.values()), "in_degree (被直接依赖数)"),
    ]
    pd.DataFrame(rows).to_csv(ensure_parent(TBL / "t20_degree_stats.csv"), index=False, encoding="utf-8")

    # ---------------- 基本拓扑
    wcc = sorted((len(c) for c in nx.weakly_connected_components(G)), reverse=True)
    scc = [len(c) for c in nx.strongly_connected_components(G) if len(c) > 1]
    n_iso = sum(1 for v in G if G.degree(v) == 0)

    deg_assort = nx.degree_assortativity_coefficient(G)

    # ---------------- 依赖深度（每个种子到其最深依赖的跳数）
    # 依赖图理论上应是 DAG；实测有 20 个小 SCC（条件依赖互相引用）。
    # 先把 SCC 凝缩成超级节点，在凝缩 DAG 上按出边方向做一次 DP，再回填。
    is_dag = nx.is_directed_acyclic_graph(G)
    depth_rows = []
    if is_dag:
        work, member_of = G, {v: v for v in G}
    else:
        C = nx.condensation(G)
        work = C
        member_of = {}
        for c, d in C.nodes(data=True):
            for m in d["members"]:
                member_of[m] = c
        log(f"图非 DAG：已在凝缩图（{C.number_of_nodes()} 个超级节点）上计算依赖深度")

    height = {}
    for v in reversed(list(nx.topological_sort(work))):
        h = 0
        for _, w in work.out_edges(v):
            h = max(h, height.get(w, 0) + 1)
        height[v] = h

    for s in seeds:
        depth_rows.append({"seed": s,
                           "dep_depth": int(height.get(member_of.get(s, s), 0)),
                           "n_direct": out_deg.get(s, 0),
                           "n_direct_dependents": in_deg.get(s, 0)})
    ddf = pd.DataFrame(depth_rows)
    ddf.to_csv(ensure_parent(TBL / "t20_depth.csv"), index=False, encoding="utf-8")

    # ---------------- 枢纽榜（被依赖最多）
    hubs = (pd.Series(in_deg, name="n_direct_dependents")
            .to_frame()
            .join(pd.Series(out_deg, name="n_direct_deps"))
            .join(nodes_frame(G))
            .sort_values("n_direct_dependents", ascending=False)
            .head(40))
    hubs.to_csv(ensure_parent(TBL / "t20_hubs.csv"), encoding="utf-8")

    # ---------------- 汇总
    struct = {
        "n_nodes": n, "n_edges": e, "n_seeds": len(seeds),
        "is_dag": bool(is_dag),
        "n_weak_components": len(wcc),
        "largest_wcc": wcc[0] if wcc else 0,
        "largest_wcc_frac": (wcc[0] / n) if n else 0.0,
        "n_isolated": n_iso,
        "n_nontrivial_scc": len(scc), "max_scc": max(scc, default=0),
        "degree_assortativity": deg_assort,
        "density": e / (n * (n - 1)) if n > 1 else 0.0,
        "dep_depth_mean": float(ddf["dep_depth"].mean()),
        "dep_depth_p50": float(ddf["dep_depth"].median()),
        "dep_depth_p95": float(ddf["dep_depth"].quantile(.95)),
        "dep_depth_max": int(ddf["dep_depth"].max()),
        "gini_in_degree": _gini(np.array(list(in_deg.values()), dtype=float)),
    }
    log(json.dumps(struct, ensure_ascii=False, indent=2))
    ensure_parent(TBL / "t20_structure.csv")
    pd.DataFrame([struct]).to_csv(TBL / "t20_structure.csv", index=False, encoding="utf-8")

    # 供下游使用：把度指标存成 parquet
    pd.DataFrame({
        "name": list(G.nodes()),
        "out_degree": [out_deg[v] for v in G.nodes()],
        "in_degree": [in_deg[v] for v in G.nodes()],
        "in_seed": [bool(G.nodes[v].get("in_seed")) for v in G.nodes()],
        "downloads": [G.nodes[v].get("downloads") for v in G.nodes()],
    }).to_parquet(ensure_parent(abspath("data/processed/degrees.parquet")), index=False)

    log("写出 t20_* 表与 degrees.parquet")
    log.close()


def nodes_frame(G: nx.DiGraph) -> pd.DataFrame:
    return pd.DataFrame(
        {"name": list(G.nodes()),
         "in_seed": [bool(G.nodes[v].get("in_seed")) for v in G.nodes()],
         "downloads": [G.nodes[v].get("downloads") for v in G.nodes()]},
    ).set_index("name")


def _gini(a: np.ndarray) -> float:
    a = np.sort(a)
    n = len(a)
    if n == 0 or a.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum()) / (n * a.sum()) - (n + 1) / n)


if __name__ == "__main__":
    main()
