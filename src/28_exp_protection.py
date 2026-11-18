"""
Step 28 —— 第三种治理动作：**保护**关键包（评审意见 C-03）

题面与分工的差异
----------------
`question.txt` 的原话是"关键包被修复 **或** 关键包被移除"——这两件事由
`src/23_exp_resilience.py` 的情景 P / R 已经做完。
但课程分工里还写了"保护"。它**不是**修复的同义词：

    修复（patch）    ：包还在、结构不变，只是它自己不再引入漏洞。
    移除（remove）   ：包从注册表消失，依赖它的东西全部构建失败。
    保护（protect）  ：包被钉版本 / vendor 进仓库 / CI 校验哈希，
                      因此**即使它自己的某个依赖坏掉，它也不会跟着坏**。

第三种动作在依赖网络上有独立含义：它**切断级联传播路径**。
一个被保护的包相当于一个"防火墙节点"：故障从一个方向撞上来就到此为止。
这正好是韧性文献里"关键节点加固"的软件版本，也解释了为什么值得单独测。

本脚本回答的问题
----------------
    在同样的破坏预算 q 下，"给 k 个枢纽包加固"能换回多少存活的种子包？
    这笔预算花在枢纽包上，比花在随机选出的包上强多少？

对照设计（每个对照排除一种具体替代解释）
----------------------------------------
  * 破坏 q 固定不变 → 排除"保护之所以有效只是因为破坏变少了"。
  * 保护额度 k 固定不变 → 排除"保护之所以有效只是因为多保护了几个"。
  * `protect_random`：随机保护同样 k 个 → 排除"任何 k 个包被保护都有这个效果，
    枢纽排名没用"。
  * `k=0` 那一行 → 排除"这个提升来自模型本身"。

产出
----
results/tables/t28_protection_curves.csv    每 (damage_mode, q, k, prot_mode) 一行
results/tables/t28_protection_summary.csv   每个 (q, k) 的枢纽/随机增益对比
results/tables/t28_protection_check.json    与位集版 AND 规则的交叉校验 + 运行记账
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (abspath, bits_to_mask, CFG, ensure_parent, Log, mask_to_bits,
                    popcount_bits, rank_by, read_csv_safe, read_universe, write_json)

log = Log("28_exp_protection")
TBL = abspath(CFG["tables"]["outdir"])
P = CFG["experiments"]["protection"]
RCFG = CFG["experiments"]["resilience"]


def load():
    nodes = read_csv_safe("data/processed/nodes.csv")
    edges = read_csv_safe("data/processed/edges.csv")
    nm = pd.read_parquet(abspath("data/processed/node_metrics.parquet"))
    bits = np.load(abspath("data/processed/reach_bits.npy"))
    rnames = json.load(open(abspath("data/processed/reach_nodes.json"), encoding="utf-8"))
    G = nx.DiGraph()
    G.add_nodes_from(nodes["name"].tolist())
    G.add_edges_from(zip(edges["src"], edges["dst"]))
    return G, nm, bits, rnames


def main() -> None:
    G, nm, bits, rnames = load()
    pos = {v: i for i, v in enumerate(rnames)}
    n = len(rnames)
    nw = bits.shape[1]

    uni = read_universe()
    seeds = [s for s in uni["project"] if s in pos]
    seed_idx = np.array([pos[s] for s in seeds], dtype=np.int64)
    log(f"图 {n} 节点 / {G.number_of_edges()} 边；图内种子 {len(seeds)}")

    # 依赖者邻接表（predecessors = 依赖 v 的包），只在图内节点上建索引
    preds: list[np.ndarray] = [[] for _ in range(n)]
    for v in rnames:
        if v in G:
            preds[pos[v]] = np.array([pos[u] for u in G.predecessors(v) if u in pos],
                                     dtype=np.int64)
    log("依赖者邻接表已构建")

    def propagate(dmg_idx: np.ndarray, immune: np.ndarray) -> np.ndarray:
        """AND 规则传播，但**被保护的包既不失效也不继续传播**。

        immune[i] = True 的节点会被完全跳过：它不被标记失败，也不把失败传给它自己的
        依赖者。这就是"防火墙节点"的语义（钉版本/vendor/CI 校验哈希）。
        """
        failed = np.zeros(n, dtype=bool)
        stack: list[int] = []
        for i in dmg_idx:
            i = int(i)
            if immune[i] or failed[i]:
                continue
            failed[i] = True
            stack.append(i)
        while stack:
            v = stack.pop()
            for u in preds[v]:
                u = int(u)
                if failed[u] or immune[u]:
                    continue
                failed[u] = True
                stack.append(u)
        return failed

    def bitset_and(dmg_idx: np.ndarray) -> np.ndarray:
        """无保护时的位集版 AND 规则（与 src/23 的 fail_and 同口径）。"""
        if len(dmg_idx) == 0:
            return np.zeros(n, dtype=bool)
        acc = bits[int(dmg_idx[0])].copy()
        for i in dmg_idx[1:]:
            acc |= bits[int(i)]
        return bits_to_mask(acc, n)

    # ---------------------------------------------------------- 交叉校验
    # 保护集为空时，逐点传播的结果必须与位集并集**完全相同**，否则两套实现有 bug。
    t0 = time.time()
    rng_chk = np.random.default_rng(12345)
    chk = []
    for trial in range(5):
        m = 300 + 100 * trial
        d = rng_chk.permutation(n)[:m]
        a = propagate(np.sort(d), np.zeros(n, dtype=bool))
        b = bitset_and(np.sort(d))
        chk.append({"n_removed": int(m), "equal": bool(np.array_equal(a, b)),
                    "n_failed_propagate": int(a.sum()), "n_failed_bitset": int(b.sum())})
    ok_all = all(c["equal"] for c in chk)
    log("无保护时的交叉校验：" + ("全部一致 ✓" if ok_all else "★不一致★") +
        json.dumps(chk, ensure_ascii=False))
    if not ok_all:
        raise SystemExit("逐点传播与位集实现不一致，停止（绝不产出错误结果）")

    # ---------------------------------------------------------- 排序
    nm_i = nm.set_index("name")
    order_crit = [x for x in rank_by({str(k): float(v) for k, v in
                                      nm_i["n_transitive_dependents"].items()}) if x in pos]
    order_in = [x for x in rank_by({str(k): float(v) for k, v in
                                    nm_i["in_degree"].items()}) if x in pos]

    qs = [q for q in P["damage_fractions"]]
    ks = [int(k) for k in P["k_values"]]
    nrep = int(P["n_random_repeats"])
    rsb = int(P["random_seed_base"])
    dmg_counts = [max(1, int(round(q * n))) for q in qs]
    n_seed = max(1, len(seed_idx))

    rows = []
    for qi, q in enumerate(qs):
        m = dmg_counts[qi]
        for dmode in ("targeted", "random"):
            reps = 1 if dmode == "targeted" else nrep
            for rep in range(reps):
                if dmode == "targeted":
                    dmg_idx = np.sort(np.array([pos[x] for x in order_in[:m]], dtype=np.int64))
                else:
                    rng = np.random.default_rng(rsb + rep)
                    dmg_idx = np.sort(rng.permutation(n)[:m].astype(np.int64))
                for k in ks:
                    if k == 0:
                        immune = np.zeros(n, dtype=bool)
                        pmode = "none"
                        failed = propagate(dmg_idx, immune)
                        rows.append({"damage_mode": dmode, "rep": rep, "q": q,
                                     "n_damaged": m, "k_protected": 0,
                                     "protect_mode": pmode,
                                     "S": float((~failed[seed_idx]).mean()),
                                     "n_failed": int(failed.sum())})
                        continue
                    for pmode in ("top_critical", "random"):
                        immune = np.zeros(n, dtype=bool)
                        if pmode == "top_critical":
                            immune[[pos[x] for x in order_crit[:k]]] = True
                        else:
                            # 随机保护 k 个：rep 兼作随机种子，保证可复现
                            rngk = np.random.default_rng(rsb + 777 + rep)
                            immune[rngk.permutation(n)[:k]] = True
                        failed = propagate(dmg_idx, immune)
                        rows.append({"damage_mode": dmode, "rep": rep, "q": q,
                                     "n_damaged": m, "k_protected": k,
                                     "protect_mode": pmode,
                                     "S": float((~failed[seed_idx]).mean()),
                                     "n_failed": int(failed.sum())})
        log(f"  q={q} 完成")

    df = pd.DataFrame(rows)
    df.to_csv(ensure_parent(TBL / "t28_protection_curves.csv"),
              index=False, encoding="utf-8")
    log(f"-> t28_protection_curves.csv ({len(df)} 行)")

    # ---------------------------------------------------------- 汇总：枢纽 vs 随机
    g = df.groupby(["damage_mode", "q", "k_protected", "protect_mode"]).agg(
        S_mean=("S", "mean"), S_std=("S", "std"), n_failed_mean=("n_failed", "mean"),
    ).reset_index()
    # 基线（k=0，无保护）用于计算 ΔS。注意必须先在 DataFrame 上 set_index 再取列，
    # 否则 Series.set_index 会把列名当成"值数组"，静默得到一个错误的索引。
    base = (g[g["protect_mode"] == "none"]
            .set_index(["damage_mode", "q"])["S_mean"])
    if base.empty:
        raise SystemExit("缺少 k=0 的基线行：请确认 config 的 protection.k_values 里包含 0")
    g["S_no_protection"] = [base.get((dm, q), np.nan)
                            for dm, q in zip(g["damage_mode"], g["q"])]
    g["delta_S"] = g["S_mean"] - g["S_no_protection"]
    g.to_csv(ensure_parent(TBL / "t28_protection_summary.csv"),
             index=False, encoding="utf-8")

    # 枢纽/随机 增益比（同一 q、同一 k）
    piv = g[g["protect_mode"] != "none"].pivot_table(
        index=["damage_mode", "q", "k_protected"], columns="protect_mode",
        values="delta_S").reset_index()
    if "top_critical" in piv and "random" in piv:
        piv["gain_ratio_top_over_random"] = (
            piv["top_critical"] / piv["random"].replace(0, np.nan))
        piv.to_csv(ensure_parent(TBL / "t28_protection_gain_ratio.csv"),
                   index=False, encoding="utf-8")
        log("枢纽保护 vs 随机保护（ΔS）:\n" + piv.round(4).to_string(index=False))

    write_json(ensure_parent(TBL / "t28_protection_check.json"), {
        "cross_check_vs_bitset": chk,
        "cross_check_all_equal": ok_all,
        "n_nodes": n, "n_seeds": len(seeds),
        "k_values": ks, "damage_fractions": qs, "n_random_repeats": nrep,
        "protection_ranking": "n_transitive_dependents（降序，同值按名称）",
        "damage_ranking_targeted": "in_degree（降序，同值按名称）",
        "runtime_s": time.time() - t0,
        "interpretation_rule":
            "若 top_critical 的 ΔS 显著大于 random 的 ΔS，说明保护预算应按枢纽排名投放；"
            "若两者接近，说明『保护谁』在这种破坏模式下不重要，结论应写成否定性结果。",
    })
    log("写出 t28_* 表")
    log.close()


if __name__ == "__main__":
    main()
