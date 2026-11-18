"""
Step 24 —— 对照实验：保度序列随机化零模型

课程要求"每个对照都应排除一种具体的替代解释"。本对照要排除的替代解释是：

    "你观察到的级联影响面重尾，只是因为度分布本身重尾；
      随便一个拥有同样度序列的网络都会这样，所以这个生态并不特殊。"

做法：**有向双边交换（double edge swap）**，精确保持每个节点的入度与出度，
      只随机化"谁连谁"。然后在新网络上重算：
        * 反向可达集（级联影响面）的分布
        * S(q) 移除曲线（随机 / 按入度靶向）
      与真实网络比较（KS 检验 + 分位数 + 曲线差异）。

为什么允许零模型出现环
----------------------
"包 v 出问题 -> 所有传递依赖它的包都出问题"这条规则不需要无环。
所以在零模型上计算反向可达集是合法的；但**OR 规则**在环上语义混乱，
我们因此只在零模型上报告 AND 规则（这正是我们主张的软件真实规则）。

产出
----
results/tables/t24_null_compare.csv     真实 vs 零模型的分布统计与检验
results/tables/t24_null_curves.csv      真实 vs 零模型的 S(q) 曲线
results/tables/t24_null_swaps.json      交换过程的质量记账
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
from common import (abspath, bits_to_mask, CFG, ensure_parent, Log, rank_by,
                    read_csv_safe, read_universe, write_json)

log = Log("24_exp_nullmodel")
TBL = abspath(CFG["tables"]["outdir"])
NCFG = CFG["experiments"]["null_model"]
RCFG = CFG["experiments"]["resilience"]


# ------------------------------------------------------------------ 零模型
def randomize_edges(edges: set[tuple[str, str]], n_swaps_attempts: int,
                    rng: np.random.Generator) -> tuple[set[tuple[str, str]], dict]:
    """有向双边交换：精确保持每个节点的入度与出度，只随机化"谁连谁"。

    ★ 可复现性要点（本项目自查时踩过）：
      遍历 Python 的 set 时元素顺序在同一进程内稳定、**跨进程不稳定**
      （字符串哈希随机化）。如果直接 `list(edges)` 建列表，
      同一 seed 在不同进程里会得到不同的交换序列，零模型结果就不可复现。
      因此这里必须 `sorted(edges)` 建立确定性的初始顺序。
    """
    edges = set(edges)
    E = sorted(edges)
    m = len(E)
    attempts = 0
    success = 0
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
                   "swaps_per_edge": success / max(1, m)}


# ------------------------------------------------------------------ 反向可达
def reverse_reach_counts(G: nx.DiGraph) -> tuple[pd.DataFrame, int]:
    """反向可达集大小：SCC 凝缩 + 沿**前向拓扑序**把依赖者的可达集向上合并。

    与实验二用的是同一套口径（这样真实网络与零模型的数字才可比）。
    对非 DAG 的零模型同样正确：环内节点必然一起被触发，凝缩成超级节点不改变可达性。
    """
    names = list(G.nodes())
    idx = {v: i for i, v in enumerate(names)}
    n = len(names)
    nw = (n + 63) // 64

    C = nx.condensation(G)
    cbits: dict[int, np.ndarray] = {}
    for c in nx.topological_sort(C):
        b = np.zeros(nw, dtype=np.uint64)
        for m in C.nodes[c]["members"]:
            b[idx[m] >> 6] |= np.uint64(1) << np.uint64(idx[m] & 63)
        for q in C.predecessors(c):          # q 依赖 c 中的节点
            b |= cbits[q]
        cbits[c] = b

    cnt = np.zeros(n, dtype=np.int64)
    for c, d in C.nodes(data=True):
        mem = d["members"]
        sz = int(bits_to_mask(cbits[c], n).sum())
        for m in mem:
            # 去掉自身：超级节点规模为 1 时减 1；规模更大时同分量伙伴互相可达，
            # 这一处是**上界近似**（本项目环规模最大 8，影响可忽略，已在局限里说明）
            cnt[idx[m]] = sz - 1
    return pd.DataFrame({"name": names, "reach": cnt}), C.number_of_nodes()


def main() -> None:
    edges_df = read_csv_safe("data/processed/edges.csv")
    nm = pd.read_parquet(abspath("data/processed/node_metrics.parquet"))
    true_edges = set(zip(edges_df["src"], edges_df["dst"]))
    nodes = set(nm["name"])
    log(f"真实边 {len(true_edges)}，节点 {len(nodes)}")

    # ---------------- 真实网络
    Gt = nx.DiGraph()
    Gt.add_nodes_from(nodes)
    Gt.add_edges_from(true_edges)
    log("真实网络反向可达集…")
    true_reach, _ = reverse_reach_counts(Gt)
    true_reach = true_reach.set_index("name")["reach"]

    # ---------------- 零模型实现
    # 沿用 reach_names 的节点顺序（= nodes.csv 顺序 = 实验四里 G.nodes() 的顺序），
    # 这样"同一 seed 的随机排列"在两个脚本里给出**完全相同**的移除集合，
    # 两条 S(q) 曲线才真正可比（此前用 sorted 顺序会导致 random 曲线不一致）。
    rnames = json.load(open(abspath("data/processed/reach_nodes.json"), encoding="utf-8"))
    names_sorted = [x for x in rnames if x in nodes]
    qs = RCFG["removal_fractions"]
    n = len(names_sorted)
    removed_counts = [max(1, int(round(q * n))) for q in qs]
    nm_i = nm.set_index("name")
    # 与实验四（src/23 的 strategy_orders）使用**完全相同**的排序规则与平局规则。
    in_deg = nm_i["in_degree"]
    order_t = rank_by({str(k): float(v) for k, v in in_deg.items()})
    rng_seed = RCFG["random_seed_base"]
    seeds = set(read_universe()["project"])
    seed_mask = np.array([bool(nm_i["in_seed"].get(x, False)) or (x in seeds)
                          for x in names_sorted], dtype=bool)
    log(f"种子掩码：{int(seed_mask.sum())} / {n}")

    rows_null = []
    curve_rows = []
    for r in range(NCFG["n_realizations"]):
        rng = np.random.default_rng(NCFG["seed_base"] + r)
        log(f"零模型实现 {r+1}/{NCFG['n_realizations']}：双边交换…")
        e, meta = randomize_edges(set(true_edges), n_swaps_attempts=8 * len(true_edges), rng=rng)
        log(f"  {json.dumps(meta)}")
        write_json(ensure_parent(TBL / f"t24_null_swaps_r{r}.json"), meta)

        Gr = nx.DiGraph()
        Gr.add_nodes_from(nodes)
        Gr.add_edges_from(e)
        reach, rounds = reverse_reach_counts(Gr)
        reach = reach.set_index("name")["reach"]

        # 分布比较
        a = true_reach.reindex(names_sorted).to_numpy(dtype=float)
        b = reach.reindex(names_sorted).to_numpy(dtype=float)
        ks = stats.ks_2samp(a, b)
        rows_null.append({
            "realization": r,
            "true_mean": a.mean(), "null_mean": b.mean(),
            "true_p99": np.percentile(a, 99), "null_p99": np.percentile(b, 99),
            "true_max": a.max(), "null_max": b.max(),
            "true_gini": _gini(a), "null_gini": _gini(b),
            "ks_stat": ks.statistic, "ks_pval": ks.pvalue,
            "spearman_rank_vs_true": stats.spearmanr(a, b).statistic,
            "sum_reach_true": a.sum(), "sum_reach_null": b.sum(),
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
                curve_rows.append({"realization": r, "mode": mode, "q": q,
                                   "S_null": float((alive & seed_mask).sum() /
                                                   max(1, seed_mask.sum())),
                                   "frac_failed": float(1 - alive.mean())})
        log(f"  零模型 {r} 完成")

    ndf = pd.DataFrame(rows_null)
    ndf.to_csv(ensure_parent(TBL / "t24_null_compare.csv"), index=False, encoding="utf-8")
    log("真实 vs 零模型（反向可达集）:\n" + ndf.round(4).to_string(index=False))

    # ---------------- 真实网络的同口径曲线（与实验四**逐位可比**）
    # random 模式按去掉 RCFG["n_random_repeats"] 个 seed 求均值，与实验四的 30 次重复完全一致；
    # by_in_degree 是确定性排序，单次即精确。
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
    out = tdf.merge(cdf.groupby(["mode", "q"]).agg(
        S_null_mean=("S_null", "mean"), S_null_std=("S_null", "std")).reset_index(),
        on=["mode", "q"], how="left")
    out.to_csv(ensure_parent(TBL / "t24_null_curves.csv"), index=False, encoding="utf-8")
    log("真实 vs 零模型 S(q):\n" + out.round(4).to_string(index=False))
    log("写出 t24_* 表")
    log.close()


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


if __name__ == "__main__":
    main()
