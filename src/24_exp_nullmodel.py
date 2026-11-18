"""
Step 24 —— 对照实验：保度序列随机化零模型（两种族）

课程要求"每个对照都应排除一种具体的替代解释"。本对照要排除的替代解释是：

    "你观察到的级联影响面重尾，只是因为度分布本身重尾；
      随便一个拥有同样度序列的网络都会这样，所以这个生态并不特殊。"

做法：**有向双边交换（double edge swap）**，精确保持每个节点的入度与出度，
      只随机化"谁连谁"。然后在新网络上重算：
        * 反向可达集（级联影响面）的分布
        * S(q) 移除曲线（随机 / 按入度靶向）
      与真实网络比较（KS 检验 + 分位数 + 曲线差异）。

★ 两个零模型族（评审意见 D-01）
--------------------------------
朴素的保度交换会**破坏 DAG 性**：真实依赖图的最大强连通分量只有 8 个节点
（19 个非平凡 SCC，占 0.33% 节点），而保度交换能造出 361–601 个节点的巨型 SCC
（占 2.3–3.8% 节点）。一个 601 节点的 SCC 意味着它的每个成员都能"反向到达"
另外 600 个成员 —— 这会**纯靠人造的环**把零模型的反向可达集抬上去，
于是"真实网络比零模型脆弱得多"这个结论里混进了两种效应：

    (1) 真实的度异质性（我们想测的）
    (2) 零模型自身的巨型环（纯人造伪影，必须剥掉）

因此本项目跑**两个族**，并且在**两个口径**上各自比较：

  族 A `degree`       ：朴素保度双边交换（允许出现环）——复现经典做法，也复现这个问题
  族 B `cycle_pres`   ：保序双边交换 —— 先按真实图的凝缩拓扑序给每个节点一个全局
                        序键，交换后两条新边都必须"序键向前"，同时把原本就在 SCC
                        内部的（可能向后的）边**冻结**不参与交换。
                        于是新图里几乎不可能长出大环，度序列仍然精确保持。

  口径 ① `node`      ：反向可达的**节点**数（经典口径，对大 SCC 敏感）
  口径 ② `condensation`：反向可达的**凝缩类**数（每个 SCC 记 1，把大 SCC 的
                        人造放大压成 1，直接检验结论是否只由巨型 SCC 造成）

两族 × 两口径 = 4 条对照线，任一结论若只在族 A 成立，就说明它其实是环伪影。

产出
----
results/tables/t24_null_compare.csv     真实 vs 零模型的分布统计与检验（含族、口径、SCC 记账）
results/tables/t24_null_curves.csv      真实 vs 零模型的 S(q) 曲线
results/tables/t24_null_swaps.json      交换过程与 SCC 规模的质量记账（两族汇总）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (abspath, bits_to_mask, CFG, ensure_parent, Log, popcount_bits,
                    rank_by, read_csv_safe, read_universe, write_json)

log = Log("24_exp_nullmodel")
TBL = abspath(CFG["tables"]["outdir"])
NCFG = CFG["experiments"]["null_model"]
RCFG = CFG["experiments"]["resilience"]


# ------------------------------------------------------------------ 零模型
def randomize_edges(edges: set[tuple[str, str]], n_swaps_attempts: int,
                    rng: np.random.Generator,
                    key: dict[str, int] | None = None) -> tuple[set[tuple[str, str]], dict]:
    """有向双边交换：精确保持每个节点的入度与出度，只随机化"谁连谁"。

    ★ 可复现性要点（本项目自查时踩过）：
      遍历 Python 的 set 时元素顺序在同一进程内稳定、**跨进程不稳定**
      （字符串哈希随机化）。如果直接 `list(edges)` 建列表，
      同一 seed 在不同进程里会得到不同的交换序列，零模型结果就不可复现。
      因此这里必须 `sorted(edges)` 建立确定性的初始顺序。

    ★ key 不为 None 时启用"保序"模式（评审 D-01）：
      只有当两条新边在全局序键上都向前时才接受交换，且原本"向后"的边
      （只可能出现在真实图的 SCC 内部）被排除在可交换池之外。
      这样新图的所有非冻结边严格单调于同一序，长环失去生成条件。
    """
    edges = set(edges)
    E = sorted(edges)
    m = len(E)
    attempts = 0
    success = 0
    rejected_backward = 0

    if key is not None:
        # 冻结"序键向后"的边（真实图 SCC 内部的边），它们不参与交换
        frozen = {e for e in E if key[e[0]] >= key[e[1]]}
        pool = [e for e in E if e not in frozen]
        if len(pool) < 2:
            return edges, {"attempts": 0, "successful_swaps": 0, "swap_rate": 0.0,
                           "swaps_per_edge": 0.0, "frozen_edges": len(frozen),
                           "rejected_backward": 0}
        mp = len(pool)
        for _ in range(n_swaps_attempts):
            attempts += 1
            i, j = rng.integers(0, mp, size=2)
            if i == j:
                continue
            a, b = pool[i]
            c, d = pool[j]
            if a == d or c == b:                      # 自环
                continue
            if key[a] >= key[d] or key[c] >= key[b]:  # 保序：新边必须向前
                rejected_backward += 1
                continue
            if (a, d) in edges or (c, b) in edges or (a, d) in frozen or (c, b) in frozen:
                continue                              # 重复边
            if (a, d) == (c, b):
                continue
            edges.discard((a, b)); edges.discard((c, d))
            edges.add((a, d)); edges.add((c, b))
            pool[i] = (a, d); pool[j] = (c, b)
            success += 1
        return edges, {"attempts": attempts, "successful_swaps": success,
                       "swap_rate": success / max(1, attempts),
                       "swaps_per_edge": success / max(1, m),
                       "frozen_edges": len(frozen),
                       "rejected_backward": rejected_backward,
                       # 保持度序列的机器校验（评审 D-01 要求"能验证的约束"）
                       "degrees_preserved": True}

    for _ in range(n_swaps_attempts):
        attempts += 1
        i, j = rng.integers(0, m, size=2)
        if i == j:
            continue
        a, b = E[i]
        c, d = E[j]
        if a == d or c == b:            # 自环
            continue
        if (a, d) in edges or (c, b) in edges:   # 重复边
            continue
        # 执行交换
        edges.discard((a, b)); edges.discard((c, d))
        edges.add((a, d)); edges.add((c, b))
        E[i] = (a, d); E[j] = (c, b)
        success += 1
    return edges, {"attempts": attempts, "successful_swaps": success,
                   "swap_rate": success / max(1, attempts),
                   "swaps_per_edge": success / max(1, m),
                   "frozen_edges": 0, "rejected_backward": 0}


def order_key(G: nx.DiGraph) -> dict[str, int]:
    """给每个节点一个**全局序键**，使大多数真实边"向前"。

    凝缩图的拓扑序给出分量之间的顺序；SCC 内部再按名称字典序分配一个连续子序号
    （因此 SCC 内部的边可能"向后" —— 它们本来就是环，会被冻结）。
    """
    C = nx.condensation(G)
    pos = {c: i for i, c in enumerate(nx.topological_sort(C))}
    key: dict[str, int] = {}
    for c, d in C.nodes(data=True):
        for j, m in enumerate(sorted(d["members"])):
            key[m] = pos[c] * (len(G) + 1) + j
    return key


# ------------------------------------------------------------------ 反向可达
def reverse_reach_counts(G: nx.DiGraph) -> tuple[pd.DataFrame, dict]:
    """反向可达集：同时给出**节点数**与**凝缩类数**两种口径。

    节点口径（`reach`）与实验二完全同口径，便于横向比较；
    凝缩口径（`reach_cond`）把每个 SCC 记 1 —— 专用于检验"零模型的巨型 SCC
    是否在替我们生产结论"（评审 D-01）。

    环内节点必然一起被触发，凝缩成超级节点不改变可达性，因此两者都是精确值
    （`reach` 对 SCC 成员数 >1 的节点是"含同分量伙伴"的口径，这一点在局限里说明）。
    """
    names = list(G.nodes())
    idx = {v: i for i, v in enumerate(names)}
    n = len(names)
    nw = (n + 63) // 64

    C = nx.condensation(G)
    cnodes = list(C.nodes())
    cid = {c: i for i, c in enumerate(cnodes)}
    nc = len(cnodes)
    nwc = (nc + 63) // 64

    cbits: dict[int, np.ndarray] = {}
    ccbits: dict[int, np.ndarray] = {}
    for c in nx.topological_sort(C):
        b = np.zeros(nw, dtype=np.uint64)
        for m in C.nodes[c]["members"]:
            b[idx[m] >> 6] |= np.uint64(1) << np.uint64(idx[m] & 63)
        cb = np.zeros(nwc, dtype=np.uint64)
        ci = cid[c]
        cb[ci >> 6] |= np.uint64(1) << np.uint64(ci & 63)
        for q in C.predecessors(c):          # q 依赖 c 中的节点
            b |= cbits[q]
            cb |= ccbits[q]
        cbits[c] = b
        ccbits[c] = cb

    cnt = np.zeros(n, dtype=np.int64)
    cnt_cond = np.zeros(n, dtype=np.int64)
    for c, d in C.nodes(data=True):
        sz = int(bits_to_mask(cbits[c], n).sum())
        sz_cond = int(popcount_bits(ccbits[c][None, :])[0])
        for m in d["members"]:
            cnt[idx[m]] = sz - 1
            cnt_cond[idx[m]] = sz_cond - 1
    scc_sizes = [len(d["members"]) for _, d in C.nodes(data=True) if len(d["members"]) > 1]
    info = {
        "n_scc_nontrivial": len(scc_sizes),
        "max_scc_size": max(scc_sizes, default=0),
        "nodes_in_nontrivial_scc": int(sum(scc_sizes)),
        "frac_nodes_in_nontrivial_scc": sum(scc_sizes) / n if n else 0.0,
    }
    return pd.DataFrame({"name": names, "reach": cnt, "reach_cond": cnt_cond}), info


def _closure_up(G: nx.DiGraph, removed: set[str]) -> set[str]:
    """反向可达闭包（AND 规则下的失败集合）：从 removed 沿"谁依赖我"方向上溯。"""
    seen = set(removed)
    stack = list(removed)
    while stack:
        v = stack.pop()
        for p in G.predecessors(v):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def _gini(a: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=float))
    n = len(a)
    if n == 0 or a.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum()) / (n * a.sum()) - (n + 1) / n)


# ------------------------------------------------------------------ 主流程
def main() -> None:
    edges_df = read_csv_safe("data/processed/edges.csv")
    nm = pd.read_parquet(abspath("data/processed/node_metrics.parquet"))
    true_edges = set(zip(edges_df["src"], edges_df["dst"]))
    nodes = set(nm["name"])
    log(f"真实边 {len(true_edges)}，节点 {len(nodes)}")

    Gt = nx.DiGraph()
    Gt.add_nodes_from(nodes)
    Gt.add_edges_from(true_edges)
    log("真实网络反向可达集…")
    true_tab, true_scc = reverse_reach_counts(Gt)
    log(f"真实图 SCC 记账：{json.dumps(true_scc, ensure_ascii=False)}")
    true_tab = true_tab.set_index("name")
    true_reach = true_tab["reach"]
    true_reach_cond = true_tab["reach_cond"]

    key = order_key(Gt)
    n_backward = sum(1 for a, b in true_edges if key[a] >= key[b])
    log(f"序键下「向后」的真实边 {n_backward} / {len(true_edges)}"
        "（应等于 SCC 内部边数，将被冻结）")

    # 沿用 reach_nodes.json 的节点顺序，保证与实验四的随机移除集合逐位一致
    rnames = json.load(open(abspath("data/processed/reach_nodes.json"), encoding="utf-8"))
    names_sorted = [x for x in rnames if x in nodes]
    qs = RCFG["removal_fractions"]
    n = len(names_sorted)
    removed_counts = [max(1, int(round(q * n))) for q in qs]
    nm_i = nm.set_index("name")
    in_deg = nm_i["in_degree"]
    order_t = rank_by({str(k): float(v) for k, v in in_deg.items()})
    rng_seed = RCFG["random_seed_base"]
    seeds = set(read_universe()["project"])
    seed_mask = np.array([bool(nm_i["in_seed"].get(x, False)) or (x in seeds)
                          for x in names_sorted], dtype=bool)
    log(f"种子掩码：{int(seed_mask.sum())} / {n}")

    families = [
        ("degree", NCFG["n_realizations"], None),
        ("cycle_pres", NCFG["cycle_preserving_realizations"], key),
    ]
    ups = int(NCFG.get("swaps_per_edge", 8))
    rows_null, curve_rows, fam_meta = [], [], []

    for fam, nreal, k in families:
        log(f"==== 零模型族 {fam}（{nreal} 次实现，{ups} 次交换/边）====")
        for r in range(nreal):
            rng = np.random.default_rng(NCFG["seed_base"] + r + (0 if fam == "degree" else 500000))
            log(f"  [{fam}] {r+1}/{nreal}：双边交换…")
            e, meta = randomize_edges(set(true_edges), n_swaps_attempts=ups * len(true_edges),
                                      rng=rng, key=k)
            Gr = nx.DiGraph()
            Gr.add_nodes_from(nodes)
            Gr.add_edges_from(e)
            # 度序列保持的机器校验（失败就立刻停，不产生"看起来正常"的错误结果）
            # ★ 必须比较 dict：networkx 的 `DegreeView.__eq__` 是**对象身份比较**，
            #   `G.in_degree() == H.in_degree()` 永远为 False —— 这个坑差点让我们
            #   误判"双边交换没有保持度序列"（本项目自查记录里的一条）。
            if dict(Gr.in_degree()) != dict(Gt.in_degree()) or \
                    dict(Gr.out_degree()) != dict(Gt.out_degree()):
                raise SystemExit(f"[{fam} r={r}] 零模型没有保持度序列，停止")
            tab, scc_info = reverse_reach_counts(Gr)
            meta.update({"family": fam, "realization": r, **{f"scc_{a}": b
                                                            for a, b in scc_info.items()}})
            # 每次实现的记账都汇总进 t24_null_swaps.json；不再散落 40 个小文件
            # （评审 D-04：结果目录也要可读，不要一堆只有脚本会读的碎片）
            fam_meta.append(meta)
            tab = tab.set_index("name")

            for frac_name, col in (("node", "reach"), ("condensation", "reach_cond")):
                a = true_reach.reindex(names_sorted).to_numpy(dtype=float) if frac_name == "node" \
                    else true_reach_cond.reindex(names_sorted).to_numpy(dtype=float)
                b = tab[col].reindex(names_sorted).to_numpy(dtype=float)
                ks = stats.ks_2samp(a, b)
                rows_null.append({
                    "family": fam, "metric": frac_name, "realization": r,
                    "true_mean": a.mean(), "null_mean": b.mean(),
                    "true_p99": np.percentile(a, 99), "null_p99": np.percentile(b, 99),
                    "true_max": a.max(), "null_max": b.max(),
                    "true_gini": _gini(a), "null_gini": _gini(b),
                    "ks_stat": ks.statistic, "ks_pval": ks.pvalue,
                    "spearman_rank_vs_true": stats.spearmanr(a, b).statistic,
                    "sum_reach_true": a.sum(), "sum_reach_null": b.sum(),
                    "null_max_scc": scc_info["max_scc_size"],
                    "null_n_scc": scc_info["n_scc_nontrivial"],
                })

            # S(q) 曲线（AND 规则：failed = 被移除集合的反向可达并集）
            for mode in ["random", "by_in_degree"]:
                for q, m in zip(qs, removed_counts):
                    if mode == "random":
                        rng2 = np.random.default_rng(rng_seed + r)
                        removed = set(rng2.permutation(names_sorted)[:m].tolist())
                    else:
                        removed = set(order_t[:m]) & nodes
                    failed = _closure_up(Gr, removed)
                    alive = np.array([x not in failed for x in names_sorted], dtype=bool)
                    curve_rows.append({"family": fam, "realization": r, "mode": mode, "q": q,
                                       "S_null": float((alive & seed_mask).sum() /
                                                       max(1, seed_mask.sum())),
                                       "frac_failed": float(1 - alive.mean())})
            log(f"    [{fam}] r={r} 完成；null_max_scc={scc_info['max_scc_size']}")
        fin = [m for m in fam_meta if m["family"] == fam]
        log(f"族 {fam} 交换记账：成功率 {np.mean([m['swap_rate'] for m in fin]):.3f}，"
            f"冻结边 {fin[0]['frozen_edges']}，"
            f"最大 SCC 中位数 {int(np.median([m['scc_max_scc_size'] for m in fin]))}")

    ndf = pd.DataFrame(rows_null)
    ndf.to_csv(ensure_parent(TBL / "t24_null_compare.csv"), index=False, encoding="utf-8")
    log("真实 vs 零模型（反向可达集）汇总:\n" + ndf.groupby(["family", "metric"]).agg(
        true_mean=("true_mean", "mean"), null_mean=("null_mean", "mean"),
        null_mean_sd=("null_mean", "std"), true_p99=("true_p99", "mean"),
        null_p99=("null_p99", "mean"), ks=("ks_stat", "mean"),
        rho=("spearman_rank_vs_true", "mean"),
        null_max_scc=("null_max_scc", "median"),
    ).round(4).to_string())

    write_json(ensure_parent(TBL / "t24_null_swaps.json"), {
        "true_scc": true_scc,
        "true_backward_edges": n_backward,
        "families": {f: [m for m in fam_meta if m["family"] == f] for f, _, _ in families},
    })

    # ---------------- 真实网络的同口径曲线（与实验四逐位可比）
    log("计算真实网络的对照曲线…")
    nrep = int(RCFG["n_random_repeats"])
    curve_true = []
    for mode in ["random", "by_in_degree"]:
        for q, m in zip(qs, removed_counts):
            if mode == "random":
                vals = []
                for rep in range(nrep):
                    rng2 = np.random.default_rng(rng_seed + rep)
                    removed = set(rng2.permutation(names_sorted)[:m].tolist())
                    failed = _closure_up(Gt, removed)
                    alive = np.array([x not in failed for x in names_sorted], dtype=bool)
                    vals.append(float((alive & seed_mask).sum() / max(1, seed_mask.sum())))
                curve_true.append({"mode": mode, "q": q, "S_true": float(np.mean(vals)),
                                   "S_true_std": float(np.std(vals)),
                                   "frac_failed": float("nan")})
            else:
                removed = set(order_t[:m]) & nodes
                failed = _closure_up(Gt, removed)
                alive = np.array([x not in failed for x in names_sorted], dtype=bool)
                curve_true.append({"mode": mode, "q": q,
                                   "S_true": float((alive & seed_mask).sum() /
                                                   max(1, seed_mask.sum())),
                                   "S_true_std": 0.0,
                                   "frac_failed": float(1 - alive.mean())})
    cdf = pd.DataFrame(curve_rows)
    tdf = pd.DataFrame(curve_true)
    agg = cdf.groupby(["family", "mode", "q"]).agg(
        S_null_mean=("S_null", "mean"), S_null_std=("S_null", "std")).reset_index()
    out = agg.merge(tdf, on=["mode", "q"], how="left")[
        ["family", "mode", "q", "S_true", "S_true_std", "S_null_mean", "S_null_std", "frac_failed"]]
    out.to_csv(ensure_parent(TBL / "t24_null_curves.csv"), index=False, encoding="utf-8")
    log("真实 vs 零模型 S(q):\n" + out.round(4).to_string(index=False))
    log("写出 t24_* 表")
    log.close()


if __name__ == "__main__":
    main()
