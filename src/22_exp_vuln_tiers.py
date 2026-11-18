"""
Step 22 —— 实验三：三级口径分解 —— 结构上界 / 反事实未修复 / 现实已修复

问题意识：为什么需要第三级？
============================
只用"结构口径 vs 版本口径"会得到一个漂亮但**解释力可疑**的数字：
    结构口径命中 131 万个 (包, 漏洞) 对，版本口径只有 1.4 万个，过近似 95×。
但这里有很大一部分是**同义反复**：deps.dev 解析到的是"今天的最新版本"，
而 OSV 里大量 advisory 是多年前就已经修好的。只要修过，今天的解析结果
自然都在修复版本之后——这不是"生态消化了风险"，而是"时间过去了"。
把这一层不加区分地算进"高估"，结论会站不住。

所以本脚本把链条拆成三级，每一级回答一个不同的问题：

  口径 S（结构上界 / naive）
      只要包 S 的传递闭包里出现包 V，就算受影响。**完全无视版本**。
      = 大多数 SBOM / 依赖扫描器报告的"受影响依赖"。

  口径 F（反事实：假设这条漏洞**至今没有被修复**）
      把 vulnerable range 改成 [introduced, ∞)，问：
      "如果上游一直没发补丁，今天会有多少包真的落在这个版本空间里？"
      = 漏洞的**传播潜力**，把"补丁发布时点"这个偶然因素剥掉。

  口径 V（现实：按 OSV 给的 [introduced, fixed) 判定）
      "今天真正还落在脆弱版本区间里的包有多少？" = 存量的**现实风险**。

于是得到两级可解释的收缩：
  S → F ：**版本推进**（依赖者自己早已越过 introduced；或结构上根本没到那个深度）
  F → V ：**补丁发布止血**（上游发了 fixed，下游随解析自然跨过去）

这把"扫描器高估"从一个笼统的抱怨，变成**可归因到两项机制**的量化结论。

产出
----
results/tables/t22_advisory_impact.csv      逐条 advisory 的三级影响面
results/tables/t22_overapprox.csv           S/V 过近似倍数分布
results/tables/t22_decomposition.csv        三级口径聚合分解与归因
results/tables/t22_worst_advisories*.csv    榜单对照
results/tables/t22_mechanism.csv            逐 advisory 的"依赖者已越过修复版本"比例
data/processed/seed_vuln_profile.parquet    每个种子包的脆弱面画像
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from packaging.version import Version

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import abspath, CFG, ensure_parent, Log, read_csv_safe, read_universe, write_json
from versions import in_intervals, osv_ranges_to_intervals, parse_version

log = Log("22_exp_vuln_cascade")
TBL = abspath(CFG["tables"]["outdir"])


def lower_bounds(ranges_json) -> tuple[list[Version], bool]:
    """取每条 ECOSYSTEM range 的 introduced 下界，并回答"是否存在未封口(无 fixed)的区间"。"""
    if isinstance(ranges_json, str):
        try:
            ranges = json.loads(ranges_json)
        except Exception:  # noqa: BLE001
            return [], False
    else:
        ranges = ranges_json or []
    los: list[Version] = []
    open_ended = False
    for rg in ranges:
        if not isinstance(rg, dict):
            continue
        if (rg.get("type") or "").upper() not in ("ECOSYSTEM", "SEMVER"):
            continue
        cur = None
        closed = False
        for ev in rg.get("events") or []:
            if "introduced" in ev:
                v = ev["introduced"]
                cur = Version("0") if str(v).strip() in ("0", "0.0.0", "") else parse_version(v)
                if cur is not None:
                    los.append(cur)
                closed = False
            elif ("fixed" in ev) or ("last_affected" in ev) or ("limit" in ev):
                closed = True
        if cur is not None and not closed:
            open_ended = True
    return los, open_ended


def main() -> None:
    cl = pd.read_parquet(abspath("data/processed/closures.parquet"))
    cl = cl[cl["relation"] != "SELF"].copy()
    osv = read_csv_safe("data/processed/osv_advisories.csv")
    uni = read_universe()
    dl = {k: float(v) for k, v in zip(uni["project"], uni["downloads"])}
    seeds = set(uni["project"])
    cl = cl[cl["seed"].isin(seeds)]
    log(f"闭包记录 {len(cl)}；advisory×package 行 {len(osv)}；种子 {len(seeds)}")

    vcache: dict[str, Version | None] = {}

    def pv(s) -> Version | None:
        s = str(s)
        if s not in vcache:
            vcache[s] = parse_version(s)
        return vcache[s]

    cl["_v"] = [pv(x) for x in cl["dep_version"]]
    n_unparsable = int(cl["_v"].isna().sum())
    log(f"闭包版本无法解析 {n_unparsable}/{len(cl)} ({n_unparsable / len(cl):.3%})"
        f" —— 计入 a_unknown，不计入受影响，也不静默丢弃")

    by_dep: dict[str, tuple[list[str], list, list[float]]] = {}
    for dep, grp in cl.groupby("dep", observed=True):
        by_dep[str(dep)] = (grp["seed"].tolist(), grp["_v"].tolist(),
                            [dl.get(s, 0.0) for s in grp["seed"]])
    log(f"有依赖者记录的包 {len(by_dep)} 个")

    struct_pairs = set(zip(cl["seed"], cl["dep"]))
    real_pairs: set[tuple[str, str]] = set()

    rows = []
    n_eval = n_unk = 0
    for r in osv.itertuples(index=False):
        pkg, adv = r.package, r.advisory_id
        iv, _bad = osv_ranges_to_intervals(r.ranges)
        los, open_ended = lower_bounds(r.ranges)
        ev_raw = getattr(r, "explicit_versions", None)
        expl: list[Version] = []
        if isinstance(ev_raw, str) and ev_raw and ev_raw != "nan":
            expl = [x for x in (pv(s) for s in ev_raw.split(";")) if x is not None]

        rec = by_dep.get(pkg)
        if rec is None:
            rows.append({
                "advisory_id": adv, "package": pkg, "a_struct": 0, "a_unfixed": 0,
                "a_version": 0, "a_unknown": 0, "expo_struct": 0.0, "expo_unfixed": 0.0,
                "expo_version": 0.0, "open_ended": bool(open_ended),
                "cvss": getattr(r, "severity_cvss", np.nan),
                "published": getattr(r, "published", None),
                "fixed": getattr(r, "fixed_versions", "")})
            continue

        sd, vs, w = rec
        lo_min = min(los) if los else None
        hitV, hitF, unk = [], [], []
        for i, v in enumerate(vs):
            n_eval += 1
            if v is None:
                unk.append(i)
                continue
            if expl and v in expl:
                hitV.append(i)
                hitF.append(i)
                continue
            res = in_intervals(v, iv)
            if res is True:
                hitV.append(i)
            elif res is None:
                unk.append(i)
            if lo_min is not None and v >= lo_min:
                hitF.append(i)
            elif lo_min is None and expl and v in expl:
                hitF.append(i)
        for i in hitV:
            real_pairs.add((sd[i], pkg))
        n_unk += len(unk)

        rows.append({
            "advisory_id": adv, "package": pkg,
            "a_struct": len(set(sd)),
            "a_unfixed": len({sd[i] for i in hitF}),
            "a_version": len({sd[i] for i in hitV}),
            "a_unknown": len({sd[i] for i in unk}),
            "expo_struct": float(sum(w)),
            "expo_unfixed": float(sum(w[i] for i in hitF)),
            "expo_version": float(sum(w[i] for i in hitV)),
            "open_ended": bool(open_ended),
            "cvss": getattr(r, "severity_cvss", np.nan),
            "published": getattr(r, "published", None),
            "fixed": getattr(r, "fixed_versions", "")})

    imp = pd.DataFrame(rows)
    imp.to_csv(ensure_parent(TBL / "t22_advisory_impact.csv"), index=False, encoding="utf-8")
    log(f"逐 advisory 判定完成：{len(imp)} 行；版本判定 {n_eval:,} 次；无法判定 {n_unk:,} 次")

    # ---------------------------------------------------------- 三级分解
    hs = imp[imp["a_struct"] > 0].copy()
    S, F, V = int(hs["a_struct"].sum()), int(hs["a_unfixed"].sum()), int(hs["a_version"].sum())
    dec = pd.DataFrame([
        {"口径": "S 结构上界（无视版本，naive 扫描）", "包-漏洞对数": S,
         "相对 S 的保留比例": 1.0, "收缩归因": "—",
         "说明": "任何传递依赖者都算"},
        {"口径": "F 反事实：漏洞至今未修复", "包-漏洞对数": F,
         "相对 S 的保留比例": F / max(1, S), "收缩归因": f"版本推进 {1 - F / max(1, S):.2%}",
         "说明": "依赖者的解析版本早已越过 introduced"},
        {"口径": "V 现实：OSV 区间 + 今日解析版本", "包-漏洞对数": V,
         "相对 S 的保留比例": V / max(1, S), "收缩归因": f"补丁止血 {1 - V / max(1, F):.2%}",
         "说明": "依赖者解析版本已越过 fixed"},
    ])
    dec.to_csv(ensure_parent(TBL / "t22_decomposition.csv"), index=False, encoding="utf-8")
    log("三级口径分解:\n" + dec.round(4).to_string(index=False))

    agg = {
        "n_advisories_total": int(len(imp)),
        "n_advisories_with_at_least_one_dependent": int(len(hs)),
        "n_advisories_open_ended": int(hs["open_ended"].sum()),
        "n_advisories_with_real_impact": int((hs["a_version"] > 0).sum()),
        "frac_advisories_zero_real_impact": round(float((hs["a_version"] == 0).mean()), 4),
        "sum_S": S, "sum_F": F, "sum_V": V,
        "ratio_S_over_V": round(S / max(1, V), 2),
        "ratio_S_over_F": round(S / max(1, F), 2),
        "ratio_F_over_V": round(F / max(1, V), 2),
        "attrib_upgrade": round(1 - F / max(1, S), 4),
        "attrib_patch": round(1 - V / max(1, F), 4),
        "expo_S": float(hs["expo_struct"].sum()),
        "expo_F": float(hs["expo_unfixed"].sum()),
        "expo_V": float(hs["expo_version"].sum()),
        "expo_ratio_S_over_V": round(float(hs["expo_struct"].sum()) /
                                     max(1.0, float(hs["expo_version"].sum())), 2),
        "struct_pairs": len(struct_pairs),
        "real_pairs": len(real_pairs),
        "pair_overapprox_ratio": round(len(struct_pairs) / max(1, len(real_pairs)), 2),
        "frac_closure_versions_unparsable": round(n_unparsable / max(1, len(cl)), 4),
    }
    log("汇总：\n" + json.dumps(agg, ensure_ascii=False, indent=2))

    live = hs[hs["open_ended"]]
    if len(live):
        log(f"未封口（至今无补丁）的 advisory {len(live)} 条：结构命中 {int(live['a_struct'].sum()):,}，"
            f"现实命中 {int(live['a_version'].sum()):,}，"
            f"过近似 {live['a_struct'].sum() / max(1, live['a_version'].sum()):.2f}×")

    # 分布
    hsv = hs[hs["a_version"] > 0].copy()
    hsv["ratio"] = hsv["a_struct"] / hsv["a_version"]
    bins = [0, 1.0001, 2, 5, 10, 50, 100, np.inf]
    lbl = ["≈1（无高估）", "1–2×", "2–5×", "5–10×", "10–50×", "50–100×", ">100×"]
    hsv["bin"] = pd.cut(hsv["ratio"], bins=bins, labels=lbl, include_lowest=True)
    dist = hsv["bin"].value_counts().reindex(lbl).fillna(0).astype(int).to_frame("n_advisories")
    dist["share"] = dist["n_advisories"] / max(1, dist["n_advisories"].sum())
    dist.to_csv(ensure_parent(TBL / "t22_overapprox.csv"), encoding="utf-8")
    log("S/V 过近似倍数分布（仅 a_version>0）:\n" + dist.to_string())

    # ---------------------------------------------------------- 机制分解
    mech = []
    for r in imp.itertuples(index=False):
        if not isinstance(r.fixed, str) or not r.fixed.strip() or r.fixed == "nan":
            continue
        rec = by_dep.get(r.package)
        if rec is None:
            continue
        sd, vs, w = rec
        fixes = [x for x in (pv(s) for s in r.fixed.split(";")) if x is not None]
        if not fixes:
            continue
        thr = max(fixes)
        known = sum(1 for v in vs if v is not None)
        ok = sum(1 for v in vs if v is not None and v >= thr)
        mech.append({"advisory_id": r.advisory_id, "package": r.package,
                     "fixed_max": str(thr), "n_dependents": len(sd),
                     "n_version_known": known,
                     "frac_already_patched": ok / known if known else np.nan})
    mdf = pd.DataFrame(mech)
    if len(mdf):
        mdf.to_csv(ensure_parent(TBL / "t22_mechanism.csv"), index=False, encoding="utf-8")
        log(f"机制分解：{len(mdf)} 条带 fixed 的 advisory；"
            f"依赖者已解析到 fixed 之后的比例：中位数 {mdf['frac_already_patched'].median():.3f}，"
            f"均值 {mdf['frac_already_patched'].mean():.3f}")

    # ---------------------------------------------------------- 榜单对照
    cols = ["advisory_id", "package", "a_struct", "a_unfixed", "a_version",
            "expo_struct", "expo_unfixed", "expo_version", "cvss", "fixed", "open_ended"]
    imp.sort_values("a_struct", ascending=False).head(25)[cols].to_csv(
        ensure_parent(TBL / "t22_worst_advisories.csv"), index=False, encoding="utf-8")
    imp.sort_values("a_version", ascending=False).head(25)[cols].to_csv(
        ensure_parent(TBL / "t22_worst_advisories_real.csv"), index=False, encoding="utf-8")

    # ---------------------------------------------------------- 种子脆弱面画像
    sp = pd.DataFrame({"seed": [p[0] for p in struct_pairs]})
    prof = sp.groupby("seed").size().rename("n_struct_vuln_deps").to_frame()
    rp = pd.DataFrame({"seed": [p[0] for p in real_pairs]})
    prof = prof.join(rp.groupby("seed").size().rename("n_real_vuln_deps"), how="outer")
    prof = prof.fillna(0).astype(int)
    prof["downloads"] = [dl.get(s, np.nan) for s in prof.index]
    prof.to_parquet(ensure_parent(abspath("data/processed/seed_vuln_profile.parquet")))
    log(f"种子脆弱面画像：{len(prof)} 个种子；结构均值 {prof['n_struct_vuln_deps'].mean():.2f}，"
        f"现实均值 {prof['n_real_vuln_deps'].mean():.2f}")

    write_json(abspath("results/tables/t22_summary.json"), agg)
    log("写出 t22_* 表")
    log.close()


if __name__ == "__main__":
    main()
