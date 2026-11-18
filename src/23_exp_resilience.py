"""
Step 23 —— 实验四：韧性情景实验（移除 vs 修复），含 AND / OR 两种失效规则

为什么要两种规则（这是本项目的一个核心论点）
============================================
食物网的"次级灭绝"是 **OR 规则（析取）**：
    捕食者只有在**全部**食物来源都消失时才灭绝。
软件依赖是 **AND 规则（合取）**：
    一个包只要**任意一个**依赖缺失/不可用，就无法构建、无法运行。

所以把食物网的级联模型直接搬到软件上会**系统性低估**软件生态的脆弱性。
本项目把两种规则放在同一张网上同时算，让这个差异成为可测量的量。

两个情景
--------
情景 R（移除 / Removal）：被移除的包从注册表消失，其所有（传递）依赖者无法构建。
情景 P（修复 / Patch）：被"修复"的包仍在，只是不再引入漏洞（从脆弱源集合中剔除）。
    → 情景 P 不改变依赖结构，只缩小"暴露源"；情景 R 同时破坏可用性。
    这两个情景对同一个 k（修复/移除包的个数）给出**不同量纲**的后果，必须分开报告。

产出
----
results/tables/t23_resilience_curves.csv   每条曲线（策略×q×规则×指标）一行
results/tables/t23_patch_vs_remove.csv      情景 P vs R 的对照
results/tables/t23_adaptive_check.csv       "每步重算排序"的敏感性（对应课程"补充协议"）
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

log = Log("23_exp_resilience")
TBL = abspath(CFG["tables"]["outdir"])
E = CFG["experiments"]["resilience"]


# ------------------------------------------------------------------ 需要的数据
def load():
    nodes = read_csv_safe("data/processed/nodes.csv")
    edges = read_csv_safe("data/processed/edges.csv")
    nm = pd.read_parquet(abspath("data/processed/node_metrics.parquet"))
    cl = pd.read_parquet(abspath("data/processed/closures.parquet"))
    uni = read_universe()
    seeds = list(uni["project"])

    G = nx.DiGraph()
    G.add_nodes_from(nodes["name"].tolist())
    G.add_edges_from(zip(edges["src"], edges["dst"]))
    cl = cl[cl["relation"] != "SELF"]
    cl = cl[cl["seed"].isin(set(seeds))]

    bits = np.load(abspath("data/processed/reach_bits.npy"))
    reach_names = json.load(open(abspath("data/processed/reach_nodes.json"), encoding="utf-8"))
    return G, nm, cl, seeds, bits, reach_names


# ------------------------------------------------------------------ 失效规则
def fail_and(removed_idx: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """AND 规则（软件真实规则）：failed = ⋃_{r∈removed} reverse_reach(r)。

    因为"任一依赖坏 -> 包坏"沿依赖链单向传播，失败集合恰为被移除集合的反向可达并集。
    """
    if len(removed_idx) == 0:
        return np.zeros(bits.shape[0], dtype=bool)
    acc = bits[removed_idx[0]].copy()
    for i in removed_idx[1:]:
        acc |= bits[i]
    return bits_to_mask(acc, bits.shape[0])


def fail_or(G: nx.DiGraph, removed: set[str], alive_names: list[str]) -> set[str]:
    """OR 规则（食物网规则）：一个包只有在**全部**直接依赖都失效后才失效。

    用"剩余存活依赖计数 + 队列"实现，O(V+E)。
    """
    idx = {v: i for i, v in enumerate(alive_names)}
    n = len(alive_names)
    n_dep_alive = np.zeros(n, dtype=np.int32)   # 还活着几个直接依赖
    failed = np.zeros(n, dtype=bool)
    for v in alive_names:
        i = idx[v]
        if v in removed:
            failed[i] = True
            continue
        n_dep_alive[i] = G.out_degree(v)
    # 节点 out_degree=0 的永远不失败（没有依赖可失去）
    from collections import deque
    q = deque(v for v in alive_names if failed[idx[v]])
    # 反向索引：依赖者列表
    while q:
        v = q.popleft()
        for dep_by in G.predecessors(v):        # dep_by 依赖 v
            j = idx.get(dep_by)
            if j is None or failed[j]:
                continue
            n_dep_alive[j] -= 1
            if n_dep_alive[j] <= 0 and G.out_degree(dep_by) > 0:
                failed[j] = True
                q.append(dep_by)
    return {alive_names[i] for i in np.flatnonzero(failed)}


# ------------------------------------------------------------------ 策略
def strategy_orders(nm: pd.DataFrame, G: nx.DiGraph, seeds: list[str]) -> dict[str, list[str]]:
    """返回 {策略名: 按优先级降序排列的节点名列表}（在**原始网络**上算一次，固定排序）。

    课程建议的"补充协议"：排序按初始网络固定，还是每次移除后重算？
    默认固定（可复现、无路径依赖），另在 src/26_exp_adaptive.py 里做"每步重算"的对照。

    所有排序统一用 `rank_by()`（同值按名称字典序打破平局），
    以保证与零模型对照实验（src/24）使用**完全相同**的移除顺序——
    否则两条 S(q) 曲线在同一点上不可比。
    """
    nodes_set = set(G.nodes())
    d = nm.set_index("name")
    series = {
        "by_in_degree": d["in_degree"],
        "by_transitive_dependents": d["n_transitive_dependents"],
        "by_pagerank_importance": d["pagerank_importance"],
        "by_betweenness": d["betweenness"],
        "by_kcore": d["kcore"],
        "by_downloads": d["downloads"].fillna(-1.0),
    }
    order: dict[str, list[str] | None] = {"random": None}
    for k, s in series.items():
        vals = {str(i): float(v) for i, v in s.items()}
        order[k] = [x for x in rank_by(vals) if x in nodes_set]
    return order


# ------------------------------------------------------------------ 主流程
def main() -> None:
    G, nm, cl, seeds, bits, reach_names = load()
    pos = {v: i for i, v in enumerate(reach_names)}
    # 种子集里可能有 7 个包在 deps.dev 上不存在（已删除/测试夹具），图里没有它们，
    # 因此所有向量化的口径都只用"图内种子"（14,993），并把这个口径写进结果。
    seeds = [s for s in seeds if s in pos]
    seed_set = set(seeds)
    n_seed = len(seeds)
    log(f"图 {G.number_of_nodes()} 节点 / {G.number_of_edges()} 边；图内种子 {n_seed}")
    seed_idx = np.array([pos[s] for s in seeds], dtype=np.int64)

    # 闭包矩阵（用于"可用性"连续指标）
    seed_pos = {s: i for i, s in enumerate(seeds)}
    cl_pos = np.array([pos.get(x, -1) for x in cl["dep"]], dtype=np.int64)
    cl_seedpos = np.array([seed_pos[x] for x in cl["seed"]], dtype=np.int64)
    keep = cl_pos >= 0
    cl_pos, cl_seedpos = cl_pos[keep], cl_seedpos[keep]
    closure_sizes = np.bincount(cl_seedpos, minlength=n_seed).astype(np.float64)

    orders = strategy_orders(nm, G, seeds)
    qs = E["removal_fractions"]
    nrep = E["n_random_repeats"]

    # 预先算好 rand 的重复索引
    removed_counts = [max(1, int(round(q * G.number_of_nodes()))) for q in qs]

    rows = []
    for strat, order in orders.items():
        reps = nrep if strat == "random" else 1
        for rep in range(reps):
            rng = np.random.default_rng(E["random_seed_base"] + rep)
            if strat == "random":
                seq = rng.permutation(list(G.nodes())).tolist()
            else:
                seq = order
            prev = 0
            for q, m in zip(qs, removed_counts):
                removed = seq[:m]
                rem_idx = np.array([pos[x] for x in removed if x in pos], dtype=np.int64)
                f_and = fail_and(rem_idx, bits)
                alive_and = ~f_and
                f_or = fail_or(G, set(removed), reach_names)
                alive_or = np.array([x not in f_or for x in reach_names], dtype=bool)

                # 指标 1：种子级"完全不可用"比例（1 - S(q)）
                seeds_alive_and = alive_and[seed_idx]
                seeds_alive_or = alive_or[seed_idx]
                # 指标 2：平均可用依赖比例（连续；分子为"仍存活的闭包成员数"）
                av_and = float(alive_and[cl_pos].sum() / max(1.0, closure_sizes.sum()))
                av_or = float(alive_or[cl_pos].sum() / max(1.0, closure_sizes.sum()))
                # 指标 3：最大弱连通分量份额（与课程地铁案例同口径，便于横向比较）
                remain = [reach_names[i] for i in np.flatnonzero(alive_and)]
                if remain:
                    sub = G.subgraph(remain)
                    lcc = max((len(c) for c in nx.weakly_connected_components(sub)), default=0)
                    lcc_frac = lcc / max(1, len(remain))
                else:
                    lcc_frac = 0.0

                rows.append({
                    "strategy": strat, "rep": rep, "q": q, "n_removed": m,
                    "S_and": float(seeds_alive_and.mean()),
                    "S_or": float(seeds_alive_or.mean()),
                    "S_anyrule_min": float(min(seeds_alive_and.mean(), seeds_alive_or.mean())),
                    "availability_and": 1.0 - av_and,
                    "availability_or": 1.0 - av_or,
                    "lcc_frac_and": lcc_frac,
                    "n_failed_and": int(f_and.sum()),
                    "n_failed_or": len(f_or),
                })
                prev = m
        log(f"  策略 {strat} 完成")

    df = pd.DataFrame(rows)
    df.to_csv(ensure_parent(TBL / "t23_resilience_curves.csv"), index=False, encoding="utf-8")
    log(f"-> t23_resilience_curves.csv ({len(df)} 行)")

    # ---------------------------------------------------------- 曲线汇总
    g = df.groupby(["strategy", "q"]).agg(
        S_and_mean=("S_and", "mean"), S_and_std=("S_and", "std"),
        S_or_mean=("S_or", "mean"), S_or_std=("S_or", "std"),
        avail_and_mean=("availability_and", "mean"),
        lcc_mean=("lcc_frac_and", "mean"),
    ).reset_index()
    g.to_csv(ensure_parent(TBL / "t23_curves_summary.csv"), index=False, encoding="utf-8")

    # 关键点：达到 S=0.5 的 q（"半衰阈值"），用插值
    def q_half(sub, col):
        """S 曲线与 0.5 的交点。

        三种情况必须分开报，否则会把"最脆弱"误读成"没有数据"：
          * 网格内穿过 0.5 -> 线性插值
          * 最小 q 处就已 ≤ 0.5 -> 返回该最小 q，并标记 instant=True（瞬间崩塌）
          * 到最大 q 仍 > 0.5 -> 返回 inf（在测得的移除预算内不会崩塌）
        """
        sub = sub.sort_values("q")
        x, y = sub["q"].to_numpy(), sub[col].to_numpy()
        if y[0] <= .5:
            return float(x[0]), True
        for i in range(1, len(x)):
            if y[i - 1] > .5 >= y[i]:
                t = (y[i - 1] - .5) / (y[i - 1] - y[i])
                return float(x[i - 1] + t * (x[i] - x[i - 1])), False
        return float("inf"), False

    half = []
    for strat, sub in g.groupby("strategy"):
        qa, ia = q_half(sub, "S_and_mean")
        qo, io = q_half(sub, "S_or_mean")
        half.append({"strategy": strat, "q_half_AND": qa, "AND_instant": ia,
                     "q_half_OR": qo, "OR_instant": io})
    hdf = pd.DataFrame(half).sort_values("q_half_AND")
    hdf.to_csv(ensure_parent(TBL / "t23_half_thresholds.csv"), index=False, encoding="utf-8")
    log("半衰阈值 q（S 掉到 0.5 所需的移除比例；越小越脆弱；instant=True = 最小 q 就已崩塌）:\n"
        + hdf.to_string(index=False))

    # ---------------------------------------------------------- 情景对照 Patch vs Remove
    # 暴露量 E(k) = Σ_{v ∈ 仍存活且仍有漏洞的包} 它的（仍存活的）依赖者数
    nm_i = nm.set_index("name")
    adv_pkgs = sorted(set(read_csv_safe("results/tables/t22_advisory_impact.csv")
                          .query("a_struct>0")["package"]))
    adv_pkgs = [p for p in adv_pkgs if p in pos]
    breadth = nm_i["n_transitive_dependents"].to_dict()

    scen = []
    for k in [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]:
        top = [x for x in orders["by_transitive_dependents"][:k]]
        # 情景 P：这 k 个包被修复 -> 它们不再作为漏洞源
        e_patch = sum(breadth.get(p, 0) for p in adv_pkgs if p not in set(top))
        # 情景 R：这 k 个包被移除 -> 依赖它们的包全部不可用，因此其暴露也随之消失
        rem_idx = np.array([pos[x] for x in top if x in pos], dtype=np.int64)
        f_and = fail_and(rem_idx, bits)
        alive_m = ~f_and
        e_remove = sum(breadth.get(p, 0) for p in adv_pkgs
                       if p not in set(top) and alive_m[pos[p]])
        seeds_alive = float(alive_m[seed_idx].mean())
        scen.append({"k": k, "exposure_patch": e_patch, "exposure_remove": e_remove,
                     "S_after_remove": seeds_alive,
                     "frac_seeds_broken": 1 - seeds_alive,
                     "exposure_patch_drop_pct": np.nan})
    sdf = pd.DataFrame(scen)
    base = sdf.loc[sdf["k"] == 0, "exposure_patch"].iloc[0]
    sdf["exposure_patch_drop_pct"] = (1 - sdf["exposure_patch"] / max(1, base)) * 100
    sdf["exposure_remove_drop_pct"] = (1 - sdf["exposure_remove"] / max(1, base)) * 100
    sdf.to_csv(ensure_parent(TBL / "t23_patch_vs_remove.csv"), index=False, encoding="utf-8")
    log("情景对照（Patch vs Remove）:\n" + sdf.to_string(index=False))

    log("写出 t23_* 表")
    log.close()


if __name__ == "__main__":
    main()
