"""
Step 10 —— 从原始抓取结果构建**有向依赖网络**，并做完整的数据质量记账。

网络定义（必须写清楚，课程要求"解释节点、边与简化假设"）
--------------------------------------------------------------
节点 v ∈ N：
  * 种子节点 S  = 下载量榜前 N 名（"人们在装什么"）
  * 叶子节点 T  = 种子包闭包里出现、但不在榜上的包（不纳入会人为切断级联路径）
边 (u -> v) ∈ E：
  * 语义 = "u 依赖 v"（依赖流方向）
  * 级联传播方向是**反向**：v 出问题 -> 沿 v -> u 传播到 u
  * 只保留**直接**依赖（relation = DIRECT / DIRECT_DECLARED），不把已解析闭包的
    间接边当作网络边（否则会把中心性全部压平）
权重（在边与节点上可选）：
  * 节点权重 w(v) = 该包在快照月的下载量；榜外包 w = 缺失（不参与加权指标）
  * 边不加权：deps.dev 不给依赖强度；我们不臆造权重，并在局限性中说明

产出
----
data/processed/nodes.csv          节点表（名称、是否种子、下载量、排名、解析版本）
data/processed/edges.csv          直接依赖边表（src=依赖者, dst=被依赖者）
data/processed/closures.parquet   完整已解析闭包（seed, dep, dep_version, relation）★版本感知级联的原料
data/processed/graph_stats.json   规模/质量记账
data/processed/universe_coverage.csv  每个种子的闭包覆盖率（用于识别被截断的种子）
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import abspath, canonical_name, CFG, ensure_parent, Log, read_universe, write_json

log = Log("10_build_graph")
G_CFG = CFG["graph"]


def load_deps() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """读取 deps.dev 原始 JSONL，返回 (节点表, 边表, 闭包表, 种子记账表, 原始计数)。

    评审意见 A-01：同名记录以最后一条为准。抓取脚本会因 `--refetch-weak` 追加新记录，
    直接逐行累加会把同一个种子算两遍（边数、闭包行数都会虚高）。
    """
    path = abspath(CFG["data_sources"]["depsdev"]["raw_jsonl"])
    fail_path = abspath(CFG["data_sources"]["depsdev"]["failed_jsonl"])
    uni = read_universe()
    dl = dict(zip(uni["project"], uni["downloads"]))
    rnk = dict(zip(uni["project"], uni["rank"]))

    # --- 先去重：按 name 保留最后一条
    by_name: dict[str, dict] = {}
    order: list[str] = []
    n_raw = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_raw += 1
            rec = json.loads(line)
            nm = rec.get("name")
            if nm is None:
                continue
            if nm not in by_name:
                order.append(nm)
            by_name[nm] = rec
    recs = [by_name[k] for k in order]
    n_dupes = n_raw - len(recs)
    if n_dupes:
        log(f"[记账] 原始 {n_raw} 行，去重后 {len(recs)} 条（{n_dupes} 条重复，保留最后一条）")

    # 网络层彻底失败、连 deps.dev 都没问到的种子（来自 failed_jsonl）
    failed_names: set[str] = set()
    if fail_path.exists():
        with open(fail_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if r.get("status") != "ok":
                    failed_names.add(canonical_name(r.get("name") or ""))

    node_rows: dict[str, dict] = {}
    edge_rows: dict[tuple[str, str], str] = {}
    closure_rows: list[dict] = []
    seed_meta: list[dict] = []

    for rec in recs:
        if rec.get("status") != "ok":
            continue
        seed = canonical_name(rec["name"])
        deps = rec.get("nodes") or []
        seed_meta.append({
            "seed": seed,
            "deps_status": rec.get("deps_status"),
            "source": rec.get("source"),
            "fallback_status": rec.get("fallback_status"),
            "resolved_version": rec.get("resolved_version"),
            # A-01 严格完整性：deps.dev 已解析 且 图级无错 且 无节点级错误
            "closure_complete_strict": bool(rec.get("closure_complete_strict", False)),
            "has_graph_error": bool(rec.get("graph_error")),
            "n_node_errors": int(rec.get("n_node_errors") or 0),
            "n_nodes": len(deps),
            "n_direct": sum(1 for n in deps if n.get("relation") in ("DIRECT", "DIRECT_DECLARED")),
        })

        # 各自节点
        node_rows.setdefault(seed, {
            "name": seed,
            "in_seed": True,
            "downloads": dl.get(seed),
            "rank": rnk.get(seed),
        })
        for nd in deps:
            nm = canonical_name(nd.get("name") or "")
            if not nm:
                continue
            rel = nd.get("relation") or "UNKNOWN"
            closure_rows.append({
                "seed": seed,
                "dep": nm,
                "dep_version": nd.get("version"),
                "relation": rel,
                "bundled": bool(nd.get("bundled")),
                "errors": ";".join(nd.get("errors") or [])[:200],
            })
            if nm not in node_rows:
                node_rows[nm] = {
                    "name": nm,
                    "in_seed": nm in dl,
                    "downloads": dl.get(nm),
                    "rank": rnk.get(nm),
                }
            # 直接依赖 -> 建边（自依赖没有网络含义，直接跳过，见评审 A-04）
            if rel in ("DIRECT", "DIRECT_DECLARED"):
                if nm == seed:
                    continue
                edge_rows[(seed, nm)] = rel

    nodes = pd.DataFrame(list(node_rows.values()))
    nodes["in_seed"] = nodes["name"].isin(set(dl))
    # 评审意见 A-04：必须显式区分"我们抓过它出边的节点"与"只是被依赖到、没抓过出边的节点"。
    # 榜外叶子在图里是**未扩展边界**：不能说它"没有依赖"，只能说"我们没问过它"。
    nodes["expansion"] = np.where(nodes["in_seed"], "seed_expanded",
                                  "boundary_not_expanded")
    edges = pd.DataFrame(
        [{"src": a, "dst": b, "rel": r} for (a, b), r in edge_rows.items()]
    )
    closure = pd.DataFrame(closure_rows)
    sm = pd.DataFrame(seed_meta)

    raw = {
        "n_lines_raw": n_raw,
        "n_records_unique": len(recs),
        "n_duplicate_lines": n_dupes,
        "n_failed_seeds": len(failed_names - {r["name"] for r in recs}),
    }
    log(f"读入 {len(recs)} 条去重后成功记录")
    log(f"节点 {len(nodes)}（其中榜内 {int(nodes['in_seed'].sum())}）")
    log(f"直接依赖边 {len(edges)}")
    log(f"闭包记录 {len(closure)}")
    return nodes, edges, closure, sm, raw


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for r in nodes.itertuples(index=False):
        G.add_node(r.name, in_seed=bool(r.in_seed), downloads=r.downloads, rank=r.rank)
    for r in edges.itertuples(index=False):
        G.add_edge(r.src, r.dst, rel=r.rel)
    return G


def main() -> None:
    nodes, edges, closure, sm, raw = load_deps()

    # ---------------------------------------------------- 图构建与结构检查
    G = build_graph(nodes, edges)
    n, e = G.number_of_nodes(), G.number_of_edges()

    # 环检测：依赖图应当是 DAG。有环说明解析结果里存在版本条件互换（少见但要如实报告）
    is_dag = nx.is_directed_acyclic_graph(G)
    cycles = []
    if not is_dag:
        try:
            cyc = nx.find_cycle(G)
            cycles = [(u, v) for u, v, *_ in cyc]
        except Exception:  # noqa: BLE001
            pass
        # 用强连通分量统计环的规模
        sccs = [c for c in nx.strongly_connected_components(G) if len(c) > 1]
        log(f"[警告] 存在环：SCC>1 的分量 {len(sccs)} 个，最大 {max((len(c) for c in sccs), default=0)}")
    else:
        sccs = []
        log("图是 DAG（无环），可以安全使用拓扑序做级联与深度分析")

    # 连通性：依赖图天然是不连通的（大量孤立叶子），报告弱连通分量分布
    wccs = sorted((len(c) for c in nx.weakly_connected_components(G)), reverse=True)
    n_iso = sum(1 for v in G if G.degree(v) == 0)

    # ---------------------------------------------------- 数据质量记账（评审 A-01）
    # 关键修正：原来把 deps_status=="ok" 直接当成"闭包完整"，这是过度声称。
    # deps.dev 返回了对象，不等于它解析成功、也不等于解析结果无错。
    # 现在分三层记账：已解析 / 严格完整 / 只有声明层回退 / 完全失败。
    n_universe = len(read_universe())
    n_seed_ok = int((sm["deps_status"] == "ok").sum())
    n_seed_unres = int((sm["deps_status"] == "unresolved").sum())
    n_strict = int(sm["closure_complete_strict"].sum())
    fb = sm["fallback_status"].fillna("")
    n_fb_ok = int((fb == "ok").sum())
    n_fb_empty = int((fb == "no_requires_dist").sum())
    n_fb_fail = int((fb == "fetch_failed").sum())
    n_graph_err = int(sm["has_graph_error"].sum())
    n_node_err_seeds = int((sm["n_node_errors"] > 0).sum())
    n_node_err_total = int(sm["n_node_errors"].sum())

    # 闭包层的真实错误/打包观测（原来这两个字段一直是空的，值得单独报出来）
    n_closure_err = int((closure["errors"].astype(str).str.len() > 0).sum()) if len(closure) else 0
    n_closure_bundled = int(closure["bundled"].sum()) if len(closure) else 0

    # 闭包被截断的种子：只有回退源（声明层）的种子，其传递闭包不完整
    truncated = set(sm.loc[~sm["closure_complete_strict"], "seed"])

    stats = {
        "snapshot_universe": str(read_universe()["snapshot"].iloc[0]),
        "n_nodes": n,
        "n_edges": e,
        "n_seed_nodes": int(nodes["in_seed"].sum()),
        "n_external_leaves": int((~nodes["in_seed"]).sum()),
        "density_directed": e / (n * (n - 1)) if n > 1 else 0.0,
        "mean_out_degree": e / n if n else 0.0,
        "n_isolated_nodes": n_iso,
        "is_dag": bool(is_dag),
        "n_cycles_scc": len(sccs),
        "max_scc_size": max((len(c) for c in sccs), default=0),
        "n_weak_components": len(wccs),
        "largest_wcc": wccs[0] if wccs else 0,
        "largest_wcc_fraction": (wccs[0] / n) if wccs and n else 0.0,
        # ---- 抓取记账
        "raw_jsonl_lines": raw["n_lines_raw"],
        "records_unique": raw["n_records_unique"],
        "duplicate_lines_deduped": raw["n_duplicate_lines"],
        "universe_seeds": n_universe,
        # ---- 种子层三分类（互斥、加总等于 records_unique + 失败数）
        "seeds_resolved_ok": n_seed_ok,
        "seeds_unresolved_fallback": n_seed_unres,
        "seeds_missing": int(raw["n_failed_seeds"]),
        # ---- 严格完整性（评审 A-01 的核心）
        "seeds_strict_complete": n_strict,
        "seeds_strict_incomplete": len(sm) - n_strict,
        "strict_complete_fraction": n_strict / len(sm) if len(sm) else 0.0,
        "seeds_with_graph_error": n_graph_err,
        "seeds_with_node_errors": n_node_err_seeds,
        "node_level_errors_total": n_node_err_total,
        # ---- 回退层细分（"抓取失败"绝不等于"没有依赖"）
        "fallback_ok": n_fb_ok,
        "fallback_no_requires_dist": n_fb_empty,
        "fallback_fetch_failed": n_fb_fail,
        # ---- 闭包层观测
        "closure_rows": len(closure),
        "closure_rows_with_error": n_closure_err,
        "closure_rows_bundled": n_closure_bundled,
        "seeds_with_truncated_closure": len(truncated),
        "frac_closure_nodes_in_universe": float(
            closure["dep"].isin(set(nodes.loc[nodes["in_seed"], "name"])).mean()
        ) if len(closure) else 0.0,
    }
    log(json.dumps(stats, ensure_ascii=False, indent=2))

    # ---------------------------------------------------- 落盘
    nodes.to_csv(ensure_parent(abspath("data/processed/nodes.csv")), index=False, encoding="utf-8")
    edges.to_csv(ensure_parent(abspath("data/processed/edges.csv")), index=False, encoding="utf-8")
    closure.to_parquet(ensure_parent(abspath("data/processed/closures.parquet")), index=False)
    sm.to_csv(ensure_parent(abspath("data/processed/universe_coverage.csv")), index=False, encoding="utf-8")
    write_json(abspath("data/processed/graph_stats.json"), stats)
    log("已写出 nodes.csv / edges.csv / closures.parquet / graph_stats.json / universe_coverage.csv")
    log.close()


if __name__ == "__main__":
    main()
