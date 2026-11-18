"""
Step 26 —— 排序更新协议敏感性（课程"补充协议"那条）

课程原话：*"排序按初始网络固定，还是每次移除后重新计算？"*
这是一个真实的建模分叉，会改变结论。本脚本把它变成一个可测的对照。

同时给出一个**结构性观察**：
  * 入度（in-degree）排序对"移除其他节点"是**不变**的——
    删掉节点 v 只会删掉指向 v 的边，不会改变任何**其他**节点的入度。
    所以 in-degree 的"固定排序"与"自适应排序"在数学上等价，无需实验即可断言。
  * 传递依赖者数、反向 PageRank、介数则都会随移除而改变（它们依赖全局结构），
    必须实测。

产出
----
results/tables/t26_adaptive.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (abspath, bits_to_mask, CFG, ensure_parent, Log, rank_by,
                    read_csv_safe, read_universe)

log = Log("26_exp_adaptive")
TBL = abspath(CFG["tables"]["outdir"])

QS = [0.005, 0.01, 0.02, 0.05, 0.10]
K_BTW = 200
ALPHA = CFG["experiments"]["bottleneck"]["reverse_pagerank_alpha"]


def load():
    nodes = read_csv_safe("data/processed/nodes.csv")
    edges = read_csv_safe("data/processed/edges.csv")
    bits = np.load(abspath("data/processed/reach_bits.npy"))
    reach_names = json.load(open(abspath("data/processed/reach_nodes.json"), encoding="utf-8"))
    seeds = set(read_universe()["project"])
    G = nx.DiGraph()
    G.add_nodes_from(nodes["name"].tolist())
    G.add_edges_from(zip(edges["src"], edges["dst"]))
    return G, bits, reach_names, seeds


def s_curve(G, bits, reach_names, seeds, ranking_provider, qs):
    """按 ranking_provider(live_G) 给出的顺序逐步移除，返回 S(q) 序列。"""
    pos = {v: i for i, v in enumerate(reach_names)}
    n = G.number_of_nodes()
    live = G.copy()
    removed_idx: list[int] = []
    out = []
    prev_k = 0
    for q in qs:
        k = max(1, int(round(q * n)))
        need = k - prev_k
        if need > 0:
            order = ranking_provider(live)
            cand = [x for x in order if x in live][:need]
            removed_idx += [pos[x] for x in cand if x in pos]
            live.remove_nodes_from(cand)
        prev_k = k
        acc = bits[removed_idx[0]].copy()
        for i in removed_idx[1:]:
            acc |= bits[i]
        alive = ~bits_to_mask(acc, len(reach_names))
        sm = np.array([x in seeds for x in reach_names])
        out.append({"q": q, "k": len(removed_idx),
                    "S": float(alive[sm].mean())})
    return out


def main() -> None:
    G, bits, reach_names, seeds = load()
    nm = pd.read_parquet(abspath("data/processed/node_metrics.parquet")).set_index("name")
    log(f"图 {G.number_of_nodes()} 节点；自适应检查 q ∈ {QS}")

    # 固定排序（原始网络）
    static = {
        "by_in_degree": lambda g: rank_by({str(k): float(v) for k, v in nm["in_degree"].items()}),
        "by_pagerank_importance": lambda g: rank_by({str(k): float(v) for k, v in nm["pagerank_importance"].items()}),
        "by_betweenness": lambda g: rank_by({str(k): float(v) for k, v in nm["betweenness"].items()}),
    }

    # 自适应排序（每步在剩余网络上重算）
    def _in_deg(g):
        return sorted(g.nodes(), key=lambda v: g.in_degree(v), reverse=True)

    def _rpr(g):
        if g.number_of_nodes() == 0:
            return []
        # 与 node_metrics 里的 pagerank_importance 保持**同一方向**：
        # PageRank 算在依赖流图 G 上（"被多少重要的包依赖"），而不是反向图。
        # 方向搞反会得到完全相反的排名——本项目在这里自查过一次。
        p = nx.pagerank(g, alpha=ALPHA, max_iter=100, tol=1e-8)
        return sorted(p, key=lambda v: p[v], reverse=True)

    def _btw(g):
        if g.number_of_nodes() < 10:
            return list(g.nodes())
        b = nx.betweenness_centrality(g, k=min(K_BTW, g.number_of_nodes()),
                                     normalized=True, seed=7)
        return sorted(b, key=lambda v: b[v], reverse=True)

    adaptive = {"by_in_degree": _in_deg, "by_pagerank_importance": _rpr, "by_betweenness": _btw}

    rows = []
    for name in static:
        log(f"  {name} · 固定排序…")
        for r in s_curve(G, bits, reach_names, seeds, static[name], QS):
            rows.append({"strategy": name, "protocol": "fixed", **r})
        log(f"  {name} · 自适应排序（每步重算）…")
        for r in s_curve(G, bits, reach_names, seeds, adaptive[name], QS):
            rows.append({"strategy": name, "protocol": "adaptive", **r})

    df = pd.DataFrame(rows)
    df.to_csv(ensure_parent(TBL / "t26_adaptive.csv"), index=False, encoding="utf-8")
    piv = df.pivot_table(index=["strategy", "q"], columns="protocol", values="S").reset_index()
    piv["delta_adaptive_minus_fixed"] = piv["adaptive"] - piv["fixed"]
    log("固定 vs 自适应（S(q)）:\n" + piv.round(4).to_string(index=False))
    piv.to_csv(ensure_parent(TBL / "t26_adaptive_pivot.csv"), index=False, encoding="utf-8")
    log("写出 t26_* 表")
    log.close()


if __name__ == "__main__":
    main()
