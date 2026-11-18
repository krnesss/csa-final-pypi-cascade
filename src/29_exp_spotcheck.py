"""
Step 29 —— 边的**逐条可核对性**：分层抽样 + 抽样核对清单（评审意见 D-03）

为什么需要它
------------
整篇结论都建立在"边 u→v 表示 u 直接依赖 v"这一条定义上。
如果边本身错了，后面所有指标都是错的，而且**指标长得很好看**——
这正是复杂网络分析最危险的失败模式：错了也看不出来。

所以必须给评审者一个可以直接核对的清单：随机抽 N 条边，
把它们连同"原始证据"（数据源里的哪条记录、上游包的被解析版本、deps.dev 链接、
以及 PyPI 的声明层原文）一起导出，让人**几十秒就能验几条**。

三层抽样（覆盖不同类型的边，而不是只抽最容易对的那种）
------------------------------------------------------
  S1 种子内部边      ：src 与 dst 都在下载量榜内（本项目最主要的边）
  S2 种子→叶子边     ：src 在榜内、dst 在榜外（榜单截断的边缘）
  S3 声明层回退边    ：rel = DIRECT_DECLARED（PyPI requires_dist 回退而来，
                        语义与"已解析依赖"不同，必须单独核对）

每层独立抽样，抽样用固定 seed，可复现。

产出
----
results/tables/t29_edge_spotcheck.csv      抽样清单（含核对用的 URL 与声明原文）
results/tables/t29_edge_spotcheck_meta.json 分层规模、抽样比例、核对结果回填位置说明
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (abspath, CFG, ensure_parent, Log, read_csv_safe, read_universe,
                    write_json)

log = Log("29_exp_spotcheck")
TBL = abspath(CFG["tables"]["outdir"])
N_PER_STRATUM = 40
SEED = 20251207


def main() -> None:
    edges = read_csv_safe("data/processed/edges.csv")
    uni = read_universe()
    seed_set = set(uni["project"])

    edges = edges.copy()
    edges["src_in_seed"] = edges["src"].isin(seed_set)
    edges["dst_in_seed"] = edges["dst"].isin(seed_set)
    # 被解析版本只对种子包有记录（data/processed/universe_coverage.csv），
    # 榜外叶子没有，因此 S2 的 check_url 会退化成"包级元数据页"，这在清单里可见。
    cov = read_csv_safe("data/processed/universe_coverage.csv")
    rv = {str(a): (None if pd.isna(b) else str(b))
          for a, b in zip(cov["seed"], cov["resolved_version"])}
    edges["src_ver"] = edges["src"].map(rv)

    strata = {
        "S1_seed_to_seed": edges[(edges["src_in_seed"]) & (edges["dst_in_seed"]) &
                                 (edges["rel"] == "DIRECT")],
        "S2_seed_to_leaf": edges[(edges["src_in_seed"]) & (~edges["dst_in_seed"]) &
                                 (edges["rel"] == "DIRECT")],
        "S3_declared_fallback": edges[edges["rel"] == "DIRECT_DECLARED"],
    }

    rng = np.random.default_rng(SEED)
    picks = []
    meta = {"seed_for_sampling": SEED, "n_per_stratum": N_PER_STRATUM, "strata": {}}
    for name, sub in strata.items():
        sub = sub.sort_values(["src", "dst"]).reset_index(drop=True)   # 确定性顺序
        n = len(sub)
        take = min(N_PER_STRATUM, n)
        idx = np.sort(rng.choice(n, size=take, replace=False)) if n else np.array([], dtype=int)
        meta["strata"][name] = {"population": int(n), "sampled": int(take),
                                "sampling_fraction": (take / n) if n else 0.0}
        for i in idx:
            r = sub.iloc[int(i)]
            picks.append({
                "stratum": name,
                "src": r["src"], "dst": r["dst"], "rel": r["rel"],
                "src_resolved_version": r.get("src_ver"),
                "src_on_download_board": bool(r["src_in_seed"]),
                "dst_on_download_board": bool(r["dst_in_seed"]),
                # 核对入口：deps.dev 的包/版本页面 + PyPI 的元数据 JSON
                "check_url_depsdev": f"https://deps.dev/pypi/{r['src']}",
                "check_url_pypi_json": (
                    f"https://pypi.org/pypi/{r['src']}/{r['src_ver']}/json"
                    if isinstance(r.get("src_ver"), str) and r.get("src_ver") else
                    f"https://pypi.org/pypi/{r['src']}/json"),
                # 报告里要求写明：是否与上游元数据一致（人工核对后填 Y/N/?）
                "verdict_manual": "",
                "note": "",
            })

    df = pd.DataFrame(picks)
    df.to_csv(ensure_parent(TBL / "t29_edge_spotcheck.csv"), index=False, encoding="utf-8")
    log(f"抽样 {len(df)} 条边：{json.dumps({k: v['sampled'] for k, v in meta['strata'].items()}, ensure_ascii=False)}")

    # ---- 一个可以自动核对的子问题：dst 是否真的在 universe/闭包里出现过
    # （不能替代人工核对语义，但能立刻抓出"图里凭空多出一条边"这类硬错误）
    closure = pd.read_parquet(abspath("data/processed/closures.parquet"))
    have = set(closure["dep"])
    for name, sub in strata.items():
        if not len(sub):
            continue
        hit = float(sub["dst"].isin(have).mean())
        meta["strata"][name]["dst_appears_in_some_closure"] = hit
        log(f"  {name}: dst 在闭包中出现过的比例 = {hit:.3f}")

    # ---- 边的重数与自环（结构性错误的自查）
    meta["structural_checks"] = {
        "n_edges": int(len(edges)),
        "n_duplicate_pairs": int(edges.duplicated(["src", "dst"]).sum()),
        "n_self_loops": int((edges["src"] == edges["dst"]).sum()),
        "n_rel_values": {str(k): int(v) for k, v in
                         edges["rel"].value_counts().items()},
        "n_src_leaf_but_has_edges": int((~edges["src_in_seed"]).sum()),
    }
    meta["manual_review_protocol"] = (
        "t29_edge_spotcheck.csv 的 verdict_manual 列由人工回填：打开 check_url_depsdev，"
        "在上游包该版本的依赖列表里查找 dst；一致填 Y，不一致填 N 并在 note 写实际值，"
        "无法判定填 ?。本文件交付的是『可被独立核对』的凭证，不是核对结论本身。")

    write_json(ensure_parent(TBL / "t29_edge_spotcheck_meta.json"), meta)
    log("结构自查: " + json.dumps(meta["structural_checks"], ensure_ascii=False))
    log("写出 t29_* 表")
    log.close()


if __name__ == "__main__":
    main()
