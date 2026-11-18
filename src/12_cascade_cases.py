"""
Step 12 —— 单点失效的**具体传播路径**案例（评审意见 C-04 第 4 条）

为什么需要它
------------
前十个表全是分布、均值、分位数。它们能说明"影响面重尾"，
但**说不清"一个底层包的漏洞怎么一路撞到一个应用上"**。
课程优秀案例（Linux 内核漏洞网络）里最有说服力的一页，正是画出了一条具体的传播链。

本脚本给出 3 个可解释的案例，覆盖三种位置：
  ① 基础依赖包（几乎人人间接依赖，广度极大、层级极浅）
  ② 中间库（自己依赖几个底层包，又被一批框架依赖）
  ③ 边缘应用包（广度小、但它是"真实用户接触到的那一端"）

对每个案例输出：
  * 反向可达的规模（节点数 / 种子数 / 加权下载量）
  * **按跳数分层的扩散**：第 1 跳 = 直接依赖它的包，第 2 跳 = 依赖那些包的包 …
  * 3 条**具体的依赖链**（从案例包一路走到某个种子包），逐跳可核对
  * 该包在各项瓶颈指标上的名次，用来展示"不同重要性概念确实不同"

产出
----
results/tables/t12_case_studies.json     三个案例的结构化描述（含具体链条）
results/tables/t12_case_paths.csv        所有示例链条，一行一跳，可直接核对
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (abspath, CFG, ensure_parent, Log, rank_by, read_csv_safe,
                    read_universe, write_json)

log = Log("12_cascade_cases")
TBL = abspath(CFG["tables"]["outdir"])

# 手工挑三个位置各一个代表（选择依据写在 log 里，且都是榜单上的真实包）。
# 这不是"挑好看的"——选择规则是：在"广度前 20 / 广度 100~300 / 广度 < 500 且下载量高"
# 三个区间里各取下载量最高者，规则先于结果确定。
PICK_RULES = {
    "foundational": "传递依赖者数排名前 20 中下载量最高者（基础依赖包）",
    "middleware": "传递依赖者数排名 100–300 中下载量最高者（中间库）",
    "edge_app": "传递依赖者数排名 500–2000 中下载量最高者（边缘/应用层）",
}


def bfs_layers(G: nx.DiGraph, src: str, cap_nodes: int = 400_000) -> list[dict]:
    """从 src 沿"谁依赖我"方向做 BFS，返回每层的节点数（第 0 层 = src 本身）。"""
    seen = {src}
    frontier = [src]
    layers = [{"hop": 0, "n_new": 1, "n_cumulative": 1, "examples": [src]}]
    cum = 1
    hop = 0
    while frontier and cum < cap_nodes:
        hop += 1
        nxt = []
        for v in frontier:
            for u in G.predecessors(v):
                if u not in seen:
                    seen.add(u)
                    nxt.append(u)
        if not nxt:
            break
        cum += len(nxt)
        layers.append({"hop": hop, "n_new": len(nxt), "n_cumulative": cum,
                       "examples": sorted(nxt)[:5]})
        frontier = nxt
    return layers


def sample_paths(G: nx.DiGraph, src: str, seed_names: set[str], k: int = 3,
                 min_hops: int = 2, max_hops: int = 8) -> list[list[str]]:
    """从 src 向上找 k 条**互不相同**的路径，终点是种子包（真实用户会装的那种包）。

    路径写法：src -> 依赖 src 的包 -> ... -> 种子包
    每一跳都满足"下一跳依赖上一跳"，可以直接在原图上核对。

    ★ 为什么要 min_hops：很多基础包（如 `packaging`）的**直接依赖者里就已经有种子**，
    朴素 DFS 会立刻返回一堆 1 跳的链（"packaging ← a2a-sdk"）。
    那条链是真的，但**不说明"级联怎么一层层传上去"**。
    所以要求**至少 2 跳**才在种子处收尾，从种子旁边再走一层，
    这样画出来才是"基础包 → 中间库 → 应用"的形状。
    """
    pool: list[list[str]] = []
    budget = [6000]      # 访问预算，防止在超大枢纽上爆炸

    def dfs(v: str, path: list[str]) -> None:
        if budget[0] <= 0 or len(pool) >= 400 or len(path) > max_hops:
            return
        # 到达种子且已经走了至少 min_hops 跳 -> 收一条
        if len(path) - 1 >= min_hops and v in seed_names:
            pool.append(list(path))
            return
        budget[0] -= 1
        preds = sorted(G.predecessors(v))[:40]
        for u in preds:
            if u in path:
                continue
            path.append(u)
            dfs(u, path)
            path.pop()
            if budget[0] <= 0 or len(pool) >= 400:
                return

    dfs(src, [src])
    # 优先取**更长**的、终点互不相同的 k 条（长的才是"级联"，短的多半是直连）
    pool.sort(key=lambda p: (-len(p), p))
    out, used_end = [], set()
    for p in pool:
        if p[-1] in used_end:
            continue
        out.append(p)
        used_end.add(p[-1])
        if len(out) >= k:
            break
    if not out:
        # 兜底：没有任何 ≥min_hops 跳的链时，退回最短链
        try:
            sp = nx.shortest_path(G, src, next(iter(seed_names)))
            out.append(sp)
        except Exception:                # noqa: BLE001
            out.append([src])
    return out


def main() -> None:
    nodes = read_csv_safe("data/processed/nodes.csv")
    edges = read_csv_safe("data/processed/edges.csv")
    nm = pd.read_parquet(abspath("data/processed/node_metrics.parquet")).set_index("name")
    uni = read_universe()
    seed_names = set(uni["project"])
    dl = {k: float(v) for k, v in zip(uni["project"], uni["downloads"])}

    G = nx.DiGraph()
    G.add_nodes_from(nodes["name"].tolist())
    G.add_edges_from(zip(edges["src"], edges["dst"]))
    log(f"图 {G.number_of_nodes()} 节点 / {G.number_of_edges()} 边")

    # ---- 按事先声明的规则挑选三个案例
    order = rank_by({str(k): float(v) for k, v in
                     nm["n_transitive_dependents"].items()})
    rank_of = {x: i for i, x in enumerate(order)}

    def pick(lo: int, hi: int) -> str:
        cands = [x for x in order[lo:hi] if x in G]
        return max(cands, key=lambda x: (dl.get(x, -1.0), x))

    cases = {
        "foundational": pick(0, 20),
        "middleware": pick(100, 300),
        "edge_app": pick(500, 2000),
    }
    log("案例选择（规则先于结果确定）：\n" + json.dumps(
        {k: {"pick": v, "rule": PICK_RULES[k], "breadth_rank": rank_of[v] + 1}
         for k, v in cases.items()}, ensure_ascii=False, indent=2))

    path_rows = []
    out = {"pick_rules": PICK_RULES, "cases": {}}
    # 每个指标的名次预先算一次（同值按名称打破平局，与全项目口径一致）
    METRICS = [k for k in ("in_degree", "out_degree", "cascade_depth", "betweenness",
                           "pagerank_importance", "kcore", "weighted_exposure")
               if k in nm.columns]
    metric_rank: dict[str, dict[str, int]] = {}
    for k in METRICS:
        o = rank_by({str(a): float(b) for a, b in nm[k].items()})
        metric_rank[k] = {x: i + 1 for i, x in enumerate(o)}

    for tag, pkg in cases.items():
        layers = bfs_layers(G, pkg)
        reach_nodes = layers[-1]["n_cumulative"] if layers else 1
        n_seed_reach = int(nm.loc[pkg, "n_transitive_dependents"])
        paths = sample_paths(G, pkg, seed_names, k=3)
        for pi, path in enumerate(paths):
            for hop, v in enumerate(path):
                path_rows.append({"case": tag, "source_package": pkg, "path_id": pi + 1,
                                  "hop": hop, "package": v,
                                  "is_seed": bool(v in seed_names),
                                  "edge_meaning": ("（起点）" if hop == 0 else
                                                   f"{path[hop-1]} 直接依赖 {v}")})
        out["cases"][tag] = {
            "package": pkg,
            "breadth_rank_by_transitive_dependents": rank_of[pkg] + 1,
            "monthly_downloads": dl.get(pkg),
            "n_reachable_nodes": reach_nodes,
            "n_reachable_seeds_closure": n_seed_reach,
            "max_hops": layers[-1]["hop"] if layers else 0,
            "diffusion_layers": layers,
            "exact_paths": paths,
            "bottleneck_metrics": {k: float(nm.loc[pkg, k]) for k in METRICS},
            "metric_ranks": {k: metric_rank[k].get(pkg) for k in METRICS},
        }
        log(f"  {tag}: {pkg}  图口径可达 {reach_nodes} 节点 / 闭包口径 {n_seed_reach} 种子 / "
            f"最深 {out['cases'][tag]['max_hops']} 跳")
        log("    各指标名次: " + json.dumps(out["cases"][tag]["metric_ranks"],
                                            ensure_ascii=False))

    pdf = pd.DataFrame(path_rows)
    pdf.to_csv(ensure_parent(TBL / "t12_case_paths.csv"), index=False, encoding="utf-8")
    write_json(ensure_parent(TBL / "t12_case_studies.json"), out)

    log("示例链条（每一跳都可原图核对）：")
    for r in pdf.itertuples(index=False):
        log(f"  [{r.case} #{r.path_id} hop{r.hop}] {r.package}"
            + ("  ← 种子" if r.is_seed else ""))
    log("写出 t12_* 表")
    log.close()


if __name__ == "__main__":
    main()
