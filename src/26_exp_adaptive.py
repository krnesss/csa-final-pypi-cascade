"""
Step 26 —— 排序更新协议敏感性（课程"补充协议"那条：排序固定还是每步重算？）

课程原话：*"排序按初始网络固定，还是每次移除后重新计算？"*
这是一个真实的建模分叉，会改变结论。本脚本把它变成一个可测的对照。

★ 修正一段**错误论证**（评审意见 B-02，P0）
--------------------------------------------
本脚本早期版本在注释里写过：

    "入度排序对『移除其他节点』是不变的——删掉 v 只会删掉指向 v 的边，
      不会改变任何其他节点的入度，所以固定与自适应数学等价。"

这是错的。图定义 `u→v` 为「u 依赖 v」，于是 `in_degree(v)` = 直接依赖 v 的包数。
删掉 u 会删掉边 `u→v`（对 u 的每一个依赖 v），因此 **v 的入度减 1**。
两节点反例（本脚本启动时会自检）：

    A→B, C→B        in_degree(B) = 2
    删除 A 后        in_degree(B) = 1     ← 变了

正确说法只能是：**在本实验采用的分阶段重排序粒度下，固定与更新的结果几乎相同**，
而且这个"几乎相同"本身是一个需要实测的经验事实，不是定理。

所以我们真的去实测三件事：
  1. 两节点反例（自检，写进 t26_protocol_check.json）；
  2. 分阶段协议（每跨过一个 q 档重算一次）下固定 vs 更新的 S(q) 差异；
  3. **严格逐点协议**（每删除一个包就重算一次），并且分别考察
     "只把主动移除的包从排名图里删掉" 与 "把级联失效的包也一起删掉" 两种口径。

在真实网络上第 3 件事的实测差别为什么很小（这是本项目的解释，也是可核查的量）：
枢纽包的**出度很小**——瓶颈榜前 79 名的平均出度只有 1.91，其中 38 个出度为 0。
删掉它们只改动 47,273 条边里的 151 条，因此入度排名的整体形状几乎不动
（Top-79 重合 78/79，全局 Spearman 0.9998）。**这是观测结果，不是先验推导。**

产出
----
results/tables/t26_adaptive.csv          每 (strategy, protocol, q) 一行
results/tables/t26_adaptive_pivot.csv    固定 vs 各协议的 ΔS
results/tables/t26_protocol_check.json   两节点反例自检 + 逐点协议的开销与重合度记账
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
                    read_csv_safe, read_universe, write_json)

log = Log("26_exp_adaptive")
TBL = abspath(CFG["tables"]["outdir"])

QS = [0.005, 0.01, 0.02, 0.05]
K_BTW = 200
ALPHA = CFG["experiments"]["bottleneck"]["reverse_pagerank_alpha"]


# ------------------------------------------------------------------ 自检（B-02）
def self_check_indegree() -> dict:
    """两节点反例：证明"删除节点不改变别人入度"是错的。"""
    G = nx.DiGraph([("A", "B"), ("C", "B")])   # A 依赖 B, C 依赖 B
    before = G.in_degree("B")
    G.remove_node("A")
    after = G.in_degree("B")
    ok = (before, after) == (2, 1)
    log(f"  [自检] A→B, C→B：删除 A 前后 B 的入度 = {before} -> {after}"
        + ("  ✓（入度确实会变）" if ok else "  ★与预期不符，停止★"))
    if not ok:
        raise SystemExit("入度反例自检失败，说明对图方向的理解与代码不一致")
    return {"graph": "A->B, C->B（A、C 依赖 B）",
            "in_degree_B_before": int(before), "in_degree_B_after_removing_A": int(after),
            "claim_invariance_is_false": True}


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


def _s_from_removed(removed_idx: list[int], bits: np.ndarray, seed_mask: np.ndarray) -> float:
    if not removed_idx:
        return 1.0
    acc = bits[removed_idx[0]].copy()
    for i in removed_idx[1:]:
        acc |= bits[i]
    return float((~bits_to_mask(acc, bits.shape[0]))[seed_mask].mean())


# ------------------------------------------------------------------ 分阶段协议
def s_curve_staged(G, bits, reach_names, seeds, ranking_provider, qs,
                   also_drop_failed: bool = False):
    """跨 q 档重算排序。also_drop_failed=True 时把级联失效节点也从排名图里删掉。"""
    pos = {v: i for i, v in enumerate(reach_names)}
    seed_mask = np.array([x in seeds for x in reach_names], dtype=bool)
    rn = np.array(reach_names, dtype=object)
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
            if also_drop_failed:
                # AND 规则的失败集合 = 被移除集合的反向可达并集（**不带**否定）
                failed = bits_to_mask(_acc(removed_idx, bits), len(reach_names))
                live.remove_nodes_from([x for x in rn[failed] if x in live])
        prev_k = k
        out.append({"q": q, "k": len(removed_idx),
                    "S": _s_from_removed(removed_idx, bits, seed_mask)})
    return out


def _acc(removed_idx: list[int], bits: np.ndarray) -> np.ndarray:
    if not removed_idx:
        return np.zeros(bits.shape[1], dtype=np.uint64)
    a = bits[removed_idx[0]].copy()
    for i in removed_idx[1:]:
        a |= bits[i]
    return a


# ------------------------------------------------------------------ 逐点协议（仅入度）
def per_node_indegree(G: nx.DiGraph, bits: np.ndarray, reach_names: list[str],
                      seeds: set[str], qs: list[float],
                      drop_cascade: bool) -> tuple[list[dict], dict, list[str]]:
    """严格逐点：每删除**一个**包就重算一次入度排序。

    用增量维护而不是每步重算全图：
        in_degree_live(v) = in_degree_original(v) - #{已删除的 u : u 依赖 v}
    每删一个 u，只需把 u 的**后继**（u 的依赖）的计数减 1，总代价 O(E)。
    drop_cascade=True 时，级联失效的包也一并从排名图里删掉并减计数。
    """
    pos = {v: i for i, v in enumerate(reach_names)}
    names = list(reach_names)
    n = len(names)
    seed_mask = np.array([x in seeds for x in reach_names], dtype=bool)
    indeg = np.array([G.in_degree(v) if v in G else 0 for v in names], dtype=np.int64)
    alive_flag = np.ones(n, dtype=bool)          # 是否还在排名图里
    succ = {v: [pos[w] for w in G.successors(v)] for v in names if v in G}

    total_k = max(1, int(round(max(qs) * G.number_of_nodes())))
    removed_idx: list[int] = []
    rows = {q: None for q in qs}
    # 名称字典序的排名（用于确定性打破平局），预先算一次
    name_rank = np.empty(n, dtype=np.int64)
    name_rank[np.argsort(np.array(names, dtype=object), kind="stable")] = np.arange(n)

    def _argmax_det() -> int:
        """取入度最大者；同值时按名称字典序 —— 与 rank_by 的平局规则一致。"""
        cand = np.flatnonzero(alive_flag)
        vals = indeg[cand]
        tied = cand[vals == vals.max()]
        return int(tied[np.argmin(name_rank[tied])])

    def _drop(i: int) -> None:
        """把节点 i 从排名图里删掉，并更新它依赖的那些包的入度计数。"""
        if not alive_flag[i]:
            return
        alive_flag[i] = False
        for w in succ.get(names[i], ()):
            if alive_flag[w]:
                indeg[w] -= 1

    for step in range(total_k):
        if not alive_flag.any():
            break
        best = _argmax_det()
        removed_idx.append(best)
        _drop(best)
        if drop_cascade:
            # AND 规则失败集合：把这些包也从排名图里删掉并减计数
            for j in np.flatnonzero(bits_to_mask(bits[best], n) & alive_flag):
                _drop(int(j))
        k_now = step + 1
        for q in qs:
            if rows[q] is None and k_now >= max(1, int(round(q * G.number_of_nodes()))):
                rows[q] = {"q": q, "k": k_now,
                           "S": _s_from_removed(removed_idx, bits, seed_mask)}
    out = [rows[q] for q in qs if rows[q] is not None]
    meta = {"steps_executed": len(removed_idx),
            "drop_cascade": drop_cascade,
            "n_edges_touched_note": "入度计数为增量维护，总代价 O(E)"}
    return out, meta, [names[i] for i in removed_idx]


# ------------------------------------------------------------------ 主流程
def main() -> None:
    chk = self_check_indegree()

    G, bits, reach_names, seeds = load()
    nm = pd.read_parquet(abspath("data/processed/node_metrics.parquet")).set_index("name")
    log(f"图 {G.number_of_nodes()} 节点；对照 q ∈ {QS}")

    static = {
        "by_in_degree": lambda g: rank_by({str(k): float(v) for k, v in nm["in_degree"].items()}),
        "by_pagerank_importance": lambda g: rank_by({str(k): float(v)
                                                     for k, v in nm["pagerank_importance"].items()}),
        "by_betweenness": lambda g: rank_by({str(k): float(v)
                                             for k, v in nm["betweenness"].items()}),
    }

    def _in_deg(g):
        return rank_by({str(v): float(g.in_degree(v)) for v in g.nodes()})

    def _rpr(g):
        if g.number_of_nodes() == 0:
            return []
        # 与 node_metrics 的 pagerank_importance 保持**同一方向**：
        # PageRank 算在依赖流图 G 上（"被多少重要的包依赖"），不是反向图。
        # 方向搞反会得到完全相反的排名 —— 本项目在这里自查过一次。
        p = nx.pagerank(g, alpha=ALPHA, max_iter=100, tol=1e-8)
        return rank_by({str(k): float(v) for k, v in p.items()})

    def _btw(g):
        if g.number_of_nodes() < 10:
            return list(g.nodes())
        b = nx.betweenness_centrality(g, k=min(K_BTW, g.number_of_nodes()),
                                      normalized=True, seed=7)
        return rank_by({str(k): float(v) for k, v in b.items()})

    adaptive = {"by_in_degree": _in_deg,
                "by_pagerank_importance": _rpr,
                "by_betweenness": _btw}

    rows = []
    for name in static:
        log(f"  {name} · fixed …")
        for r in s_curve_staged(G, bits, reach_names, seeds, static[name], QS):
            rows.append({"strategy": name, "protocol": "fixed", **r})
        log(f"  {name} · adaptive_staged …")
        for r in s_curve_staged(G, bits, reach_names, seeds, adaptive[name], QS):
            rows.append({"strategy": name, "protocol": "adaptive_staged", **r})
        log(f"  {name} · adaptive_staged_with_cascade …")
        for r in s_curve_staged(G, bits, reach_names, seeds, adaptive[name], QS,
                                also_drop_failed=True):
            rows.append({"strategy": name, "protocol": "adaptive_staged_with_cascade", **r})

    # 严格逐点（只对入度做，其余指标的逐点重算成本不可接受，这一点在报告里写明）
    log("  by_in_degree · per_node（严格逐点重算）…")
    r1, m1, ord1 = per_node_indegree(G, bits, reach_names, seeds, QS, drop_cascade=False)
    for r in r1:
        rows.append({"strategy": "by_in_degree", "protocol": "adaptive_per_node", **r})
    log("  by_in_degree · per_node_with_cascade（严格逐点 + 级联失效也移出排名图）…")
    r2, m2, ord2 = per_node_indegree(G, bits, reach_names, seeds, QS, drop_cascade=True)
    for r in r2:
        rows.append({"strategy": "by_in_degree",
                     "protocol": "adaptive_per_node_with_cascade", **r})

    df = pd.DataFrame(rows)
    df.to_csv(ensure_parent(TBL / "t26_adaptive.csv"), index=False, encoding="utf-8")

    piv = df.pivot_table(index=["strategy", "q"], columns="protocol",
                         values="S").reset_index()
    base_col = "fixed"
    for c in piv.columns:
        if c in ("strategy", "q", base_col):
            continue
        piv[f"delta_{c}_minus_fixed"] = piv[c] - piv[base_col]
    log("固定 vs 各协议的 S(q):\n" + piv.round(6).to_string(index=False))
    piv.to_csv(ensure_parent(TBL / "t26_adaptive_pivot.csv"), index=False, encoding="utf-8")

    # 排名重合度：固定排序 vs 逐点自适应排序（直接复用上面已经跑出来的顺序）
    static_order = static["by_in_degree"](G)
    final_orders = {False: ord1, True: ord2}

    def top_overlap(a, b, kk):
        sa, sb = set(a[:kk]), set(b[:kk])
        return len(sa & sb) / max(1, len(sa | sb)), len(sa & sb)

    ov = {}
    for kk in (10, 20, 50, 79, 100, 200):
        for dc, label in ((False, "per_node"), (True, "per_node_with_cascade")):
            jac, hit = top_overlap(static_order, final_orders[dc], kk)
            ov[f"top{kk}_{label}"] = {"jaccard": jac, "hits": hit, "k": kk}
    log("固定排序 vs 逐点自适应排序 的 Top-k 重合：")
    for kk in (10, 20, 50, 79, 100, 200):
        a = ov[f"top{kk}_per_node"]
        b = ov[f"top{kk}_per_node_with_cascade"]
        log(f"  Top-{kk}: 仅主动移除 {a['hits']}/{kk} (J={a['jaccard']:.3f})；"
            f"含级联失效 {b['hits']}/{kk} (J={b['jaccard']:.3f})")

    # 出度解释：为什么在真实网络上这个差别很小
    top79 = static_order[:79]
    outdeg = np.array([G.out_degree(v) for v in top79], dtype=float)
    touched = int(outdeg.sum())
    m_total = G.number_of_edges()
    log(f"瓶颈榜前 79 名的出度：均值 {outdeg.mean():.2f}，其中 {int((outdeg == 0).sum())} 个为 0；"
        f"删除它们共触碰 {touched} / {m_total} 条边（{100*touched/m_total:.3f}%）")

    write_json(ensure_parent(TBL / "t26_protocol_check.json"), {
        "self_check_indegree": chk,
        "protocols": ["fixed", "adaptive_staged", "adaptive_staged_with_cascade",
                      "adaptive_per_node", "adaptive_per_node_with_cascade"],
        "note_protocol_scope":
            "adaptive_per_node 只对 by_in_degree 实现：其它指标每删一个包就要重算 "
            "PageRank/介数，成本不可接受；这一点在报告里明说，不假装已做。",
        "per_node_meta": {"active_only": m1, "with_cascade": m2},
        "top_overlap_fixed_vs_per_node": ov,
        "why_difference_is_small_measured": {
            "top79_mean_out_degree": float(outdeg.mean()),
            "top79_zero_out_degree": int((outdeg == 0).sum()),
            "edges_touched_by_removing_top79": touched,
            "n_edges_total": int(m_total),
            "frac_edges_touched": 100 * touched / max(1, m_total),
        },
        "wording_rule": "结论只能写成『在本样本与本节协议粒度下观察到的差异』，"
                        "不得写成『固定与自适应等价』的数学断言。",
    })
    log("写出 t26_* 表")
    log.close()


if __name__ == "__main__":
    main()
