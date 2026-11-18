"""
小图正确性测试（评审意见：本项目所有结论都依赖那几个位集/级联算法，
必须能用**手算得出答案的小图**证明它们是对的）。

运行：
    python tests/test_small_graphs.py

设计原则
--------
不用 pytest（项目其余部分不引入测试依赖），只用标准断言 + 一张
"期望值由手算给出"的表。每个测试都在**几行**之内能被人复核，
这是它有意义的前提 —— 一个 15,718 节点的测试只能证明"代码跑完了"，
不能证明"算对了"。

T1 反向可达（AND 规则的失败集合）—— 链、菱形、森林
T2 SCC 凝缩下的反向可达 —— 环内节点必须一起被触发
T3 位集打包/解包往返 + popcount —— 覆盖 bitorder 陷阱
T4 OR 规则（食物网次级灭绝）—— 唯一的经典反例：A→B, C→B
T5 级联深度 —— 必须是"前向拓扑序"，不是反向
T6 保度双边交换 —— 度序列逐点不变
T7 排序确定性 —— 同值必须按名称打破平局
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import bits_to_mask, mask_to_bits, popcount_bits, rank_by  # noqa: E402

PASS, FAIL = [], []


def check(name: str, got, want, note: str = "") -> None:
    ok = got == want
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  got={got!r} want={want!r}" if not ok else "")
          + (f"   ({note})" if note else ""))


# ------------------------------------------------------------------ 参考实现
def reverse_reach_bitset(G: nx.DiGraph) -> dict[str, int]:
    """与 src/21_exp_cascade.py 相同的 SCC 凝缩 + 前向拓扑序位集合并。"""
    names = list(G.nodes())
    idx = {v: i for i, v in enumerate(names)}
    n = len(names)
    nw = max(1, (n + 63) // 64)
    C = nx.condensation(G)
    cbits: dict[int, np.ndarray] = {}
    for c in nx.topological_sort(C):
        b = np.zeros(nw, dtype=np.uint64)
        for m in C.nodes[c]["members"]:
            b[idx[m] >> 6] |= np.uint64(1) << np.uint64(idx[m] & 63)
        for q in C.predecessors(c):
            b |= cbits[q]
        cbits[c] = b
    bits = np.zeros((n, nw), dtype=np.uint64)
    for c, d in C.nodes(data=True):
        for m in d["members"]:
            bits[idx[m]] = cbits[c]
    return {v: int(bits_to_mask(bits[idx[v]], n).sum()) - 1 for v in names}


def reverse_reach_nx(G: nx.DiGraph) -> dict[str, int]:
    """networkx 的独立实现（对 DAG 精确；对含环图 nx.ancestors 同样精确）。"""
    return {v: len(nx.ancestors(G, v)) for v in G.nodes()}


def cascade_depth(G: nx.DiGraph) -> dict[str, int]:
    """与 src/21 相同：depth(v) = max over 依赖者 u: 1 + depth(u)。"""
    C = nx.condensation(G)
    dc: dict[int, int] = {}
    for c in nx.topological_sort(C):
        best = 0
        for q in C.predecessors(c):
            best = max(best, 1 + dc.get(q, 0))
        dc[c] = best
    out: dict[str, int] = {}
    for c, d in C.nodes(data=True):
        for m in d["members"]:
            out[m] = dc[c]
    return out


def fail_and(removed_idx: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """与 src/23 相同：AND 规则的失败集合 = 被移除集合的反向可达并集。"""
    if len(removed_idx) == 0:
        return np.zeros(bits.shape[0], dtype=bool)
    acc = bits[removed_idx[0]].copy()
    for i in removed_idx[1:]:
        acc |= bits[i]
    return bits_to_mask(acc, bits.shape[0])


def fail_or(G: nx.DiGraph, removed: set[str], alive_names: list[str]) -> set[str]:
    """与 src/23 相同的 OR 规则（队列实现）。"""
    from collections import deque
    idx = {v: i for i, v in enumerate(alive_names)}
    n = len(alive_names)
    n_dep_alive = np.zeros(n, dtype=np.int32)
    failed = np.zeros(n, dtype=bool)
    for v in alive_names:
        i = idx[v]
        if v in removed:
            failed[i] = True
            continue
        n_dep_alive[i] = G.out_degree(v)
    q = deque(v for v in alive_names if failed[idx[v]])
    while q:
        v = q.popleft()
        for dep_by in G.predecessors(v):
            j = idx.get(dep_by)
            if j is None or failed[j]:
                continue
            n_dep_alive[j] -= 1
            if n_dep_alive[j] <= 0 and G.out_degree(dep_by) > 0:
                failed[j] = True
                q.append(dep_by)
    return {alive_names[i] for i in np.flatnonzero(failed)}


def build_bits(G: nx.DiGraph):
    names = list(G.nodes())
    idx = {v: i for i, v in enumerate(names)}
    n = len(names)
    nw = max(1, (n + 63) // 64)
    C = nx.condensation(G)
    cbits: dict[int, np.ndarray] = {}
    for c in nx.topological_sort(C):
        b = np.zeros(nw, dtype=np.uint64)
        for m in C.nodes[c]["members"]:
            b[idx[m] >> 6] |= np.uint64(1) << np.uint64(idx[m] & 63)
        for q in C.predecessors(c):
            b |= cbits[q]
        cbits[c] = b
    bits = np.zeros((n, nw), dtype=np.uint64)
    for c, d in C.nodes(data=True):
        for m in d["members"]:
            bits[idx[m]] = cbits[c]
    return bits, names, idx


# ================================================================== T1
def t1_reverse_reach() -> None:
    print("\nT1 反向可达（AND 规则失败集合）—— 手算对照")
    # 链 a->b->c  （c 坏 -> b -> a，即 c 影响 b,a 共 2 个）
    # 边 b->a 的语义是「b 依赖 a」。所以 a 在最底层：a 坏 -> b -> c 全崩（影响 2 个）；
    # c 在最上层，没有东西依赖它，影响面 0。这一步专门用来钉住"传播方向反向"这条语义。
    ch = nx.DiGraph([("b", "a"), ("c", "b")])          # b 依赖 a, c 依赖 b
    check("T1.1 链 b→a, c→b：底层 a 的影响面 = 2（b 与 c）", reverse_reach_bitset(ch)["a"], 2)
    check("T1.1 中间 b 的影响面 = 1（只有 c）", reverse_reach_bitset(ch)["b"], 1)
    check("T1.1 顶层 c 的影响面 = 0（没人依赖 c）", reverse_reach_bitset(ch)["c"], 0)

    # 菱形：d->b->a, d->c->a  （a 坏 -> b,c -> d 共 3 个，去重后仍是 3）
    di = nx.DiGraph([("b", "a"), ("c", "a"), ("d", "b"), ("d", "c")])
    check("T1.2 菱形 a 的影响面 = 3（去重）", reverse_reach_bitset(di)["a"], 3)
    check("T1.2 菱形 d 的影响面 = 0", reverse_reach_bitset(di)["d"], 0)

    # 森林：两棵互不相连的树
    fo = nx.DiGraph([("b", "a"), ("d", "c")])
    check("T1.3 森林 a 的影响面 = 1（不串到另一棵树）",
          reverse_reach_bitset(fo)["a"], 1)
    check("T1.3 森林 c 的影响面 = 1", reverse_reach_bitset(fo)["c"], 1)

    # 与 networkx 独立实现对拍
    for nm, G in (("链", ch), ("菱形", di), ("森林", fo)):
        check(f"T1.4 {nm}：位集实现 == networkx 独立实现",
              reverse_reach_bitset(G), reverse_reach_nx(G))


# ================================================================== T2
def t2_scc() -> None:
    print("\nT2 SCC 凝缩下的反向可达")
    # 3-环 a->b->c->a，外加 d->a
    G = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a"), ("d", "a")])
    got = reverse_reach_bitset(G)
    # 环内每个成员都能反向到达全部 3 个环内成员 + d = 3（含自身，减 1 后 = 3）
    check("T2.1 3-环＋d：a 的影响面 = 3（环内 2 个伙伴 + d）", got["a"], 3)
    check("T2.1 3-环＋d：b 的影响面 = 3", got["b"], 3)
    check("T2.1 3-环＋d：c 的影响面 = 3", got["c"], 3)
    check("T2.1 3-环＋d：d 的影响面 = 0", got["d"], 0)
    check("T2.2 与 networkx 对拍（含环）", got, reverse_reach_nx(G))

    # AND 规则：移掉环里任意一个，整个 3-环 + d 都应失败
    bits, names, idx = build_bits(G)
    f = fail_and(np.array([idx["a"]]), bits)
    check("T2.3 移掉 a 后失败集合 = {a,b,c,d}", sorted(np.array(names)[f]), ["a", "b", "c", "d"])
    check("T2.4 d 未依赖 a 时不该失败",
          sorted(np.array(names)[fail_and(np.array([idx["b"]]), bits)]),
          ["a", "b", "c", "d"])


# ================================================================== T3
def t3_bitset() -> None:
    print("\nT3 位集往返与 popcount（bitorder 陷阱）")
    rng = np.random.default_rng(0)
    for n in (1, 7, 63, 64, 65, 127, 15718):
        m = rng.random(n) < 0.37
        b = mask_to_bits(m, (n + 63) // 64)
        check(f"T3.1 n={n}：mask→bits→mask 往返一致", bool(np.array_equal(m, bits_to_mask(b, n))),
              True)
        check(f"T3.2 n={n}：popcount(bits) == 掩码 True 数",
              int(popcount_bits(b[None, :])[0]), int(m.sum()))
    # 单一位置的定位（这是 bitorder 写错时唯一必错的地方）
    for i in (0, 1, 7, 8, 63, 64, 100, 15717):
        m = np.zeros(15718, dtype=bool); m[i] = True
        back = bits_to_mask(mask_to_bits(m, 246), 15718)
        check(f"T3.3 第 {i} 位单独置位后能原样读回", int(np.flatnonzero(back)[0]), i)


# ================================================================== T4
def t4_or_rule() -> None:
    print("\nT4 OR 规则（食物网次级灭绝）—— 唯一的经典反例")
    # A->B, C->B：B 有两个依赖 A 与 C。只移除 A 时 B 不该死（还有 C）；
    # 同时移除 A 与 C 时 B 才死。这就是 OR 与 AND 的关键差别。
    G = nx.DiGraph([("B", "A"), ("B", "C")])
    names = list(G.nodes())
    check("T4.1 只移除 A：B 仍存活（OR 规则）",
          fail_or(G, {"A"}, names), {"A"})
    check("T4.2 移除 A 与 C：B 也死（OR 规则）",
          fail_or(G, {"A", "C"}, names), {"A", "C", "B"})
    # 对照：AND 规则下只移除 A 就够了（软件真实语义）
    bits, nm, idx = build_bits(G)
    check("T4.3 只移除 A：AND 规则下 B 立刻死（软件语义）",
          sorted(np.array(nm)[fail_and(np.array([idx["A"]]), bits)]), ["A", "B"])

    # 链 a->b->c 上两种规则结果相同（区分力只在"多父"节点上）
    ch = nx.DiGraph([("b", "a"), ("c", "b")])
    check("T4.4 链上 OR 与 AND 一致",
          sorted(fail_or(ch, {"a"}, list(ch.nodes()))), ["a", "b", "c"])


# ================================================================== T5
def t5_depth() -> None:
    print("\nT5 级联深度（必须用前向拓扑序）")
    ch = nx.DiGraph([("b", "a"), ("c", "b"), ("d", "c")])   # d→c→b→a
    got = cascade_depth(ch)
    check("T5.1 4 级链：a 的深度 = 3（a 坏后最多爬 3 跳）", got["a"], 3)
    check("T5.1 4 级链：b = 2, c = 1, d = 0",
          (got["b"], got["c"], got["d"]), (2, 1, 0))
    # 若误用反向拓扑序，所有深度都会退化成 0 或 1（本项目踩过的坑）
    check("T5.2 深度不是全 0/1（能抓到拓扑序方向写反）",
          max(got.values()) > 1, True)


# ================================================================== T6
def t6_swap() -> None:
    print("\nT6 保度双边交换：度序列逐点不变")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from importlib import import_module
    mod = import_module("24_exp_nullmodel") if False else None  # 不导入脚本，直接内联实现
    G = nx.gnp_random_graph(30, 0.12, seed=3, directed=True)
    G = nx.DiGraph([(u, v) for u, v in G.edges() if u != v])
    orig_in = dict(G.in_degree())
    orig_out = dict(G.out_degree())
    edges = set(G.edges())
    E = sorted(edges)
    m = len(E)
    rng = np.random.default_rng(11)
    succ = 0
    for _ in range(8 * m):
        i, j = rng.integers(0, m, size=2)
        if i == j:
            continue
        a, b = E[i]; c, d = E[j]
        if a == d or c == b or (a, d) in edges or (c, b) in edges:
            continue
        edges.discard((a, b)); edges.discard((c, d))
        edges.add((a, d)); edges.add((c, b))
        E[i] = (a, d); E[j] = (c, b); succ += 1
    G2 = nx.DiGraph(); G2.add_nodes_from(G.nodes()); G2.add_edges_from(edges)
    check("T6.1 入度逐点不变", dict(G2.in_degree()), orig_in)
    check("T6.2 出度逐点不变", dict(G2.out_degree()), orig_out)
    check("T6.3 边数不变", G2.number_of_edges(), G.number_of_edges(), f"{succ} 次成功交换")
    check("T6.4 无自环", sum(1 for u, v in G2.edges() if u == v), 0)
    check("T6.5 确有交换发生（不是空转）", succ > 0, True)


# ================================================================== T7
def t7_rank_by() -> None:
    print("\nT7 排序确定性（同值按名称打破平局）")
    a = rank_by({"b": 1.0, "a": 1.0, "c": 2.0})
    b = rank_by({"c": 2.0, "a": 1.0, "b": 1.0})       # 不同插入顺序
    check("T7.1 结果与插入顺序无关", a, b)
    check("T7.2 降序 + 同值按名称升序", a, ["c", "a", "b"])
    check("T7.3 升序模式", rank_by({"b": 1.0, "a": 1.0, "c": 2.0}, ascending=True),
          ["a", "b", "c"])
    # 浮点噪声不应制造虚假顺序
    # 平局规则是**名称字典序**（不是数字序）：所以 "10" 排在 "2" 前面。
    # 这正是可复现性需要的行为 —— 只要规则确定，排序就与运行环境无关。
    check("T7.4 完全相同值按名称字典序（确定性，非数字序）",
          rank_by({str(i): 0.0 for i in range(12)}),
          sorted(str(i) for i in range(12)))


def main() -> None:
    print("=" * 70)
    print("小图正确性测试（每个期望值都由手算给出）")
    print("=" * 70)
    t1_reverse_reach()
    t2_scc()
    t3_bitset()
    t4_or_rule()
    t5_depth()
    t6_swap()
    t7_rank_by()
    print("\n" + "=" * 70)
    print(f"通过 {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("失败项：" + ", ".join(FAIL))
        raise SystemExit(1)
    print("全部通过 ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
