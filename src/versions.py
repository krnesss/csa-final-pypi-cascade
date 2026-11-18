"""
版本语义工具：把 OSV 的 ranges/versions 转成可判定的区间集合。

为什么需要一个专门的模块
------------------------
"版本约束感知的级联"（实验三）的整个结论都依赖于
  v_D(P) ∈ vulnerable(D) ?
这一步判定。它必须：
  1) 用 PEP 440 的版本序关系（`1.0.0rc1 < 1.0.0`，`1.0.0.dev0 < 1.0.0`），不能字符串比较；
  2) 正确处理 OSV 的 5 种 event：introduced / fixed / last_affected / limit / 显式 versions；
  3) 对无法解析的版本**显式记为 unknown**，而不是默认"不受影响"——
     默认 False 会系统性低估影响，默认 True 会系统性高估。我们要求两者都能被报告。
"""
from __future__ import annotations

import json
from typing import Any, Iterable, List, Optional, Tuple

from packaging.version import InvalidVersion, Version

EPS = Version("0")


def parse_version(s: Any) -> Optional[Version]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return Version(s)
    except InvalidVersion:
        # 常见噪声：尾随空白、`v` 前缀、`==1.2.3`
        t = s.lstrip("vV= ").strip()
        try:
            return Version(t)
        except InvalidVersion:
            return None


# 区间用 (lo, lo_incl, hi, hi_incl) 表示；hi=None 表示无穷
Interval = Tuple[Optional[Version], bool, Optional[Version], bool]


def osv_ranges_to_intervals(ranges_json: str | list) -> Tuple[List[Interval], int]:
    """把 OSV 的 ranges 字段转成区间列表。

    返回 (intervals, n_unparsable_events)。无法解析的边界会被跳过并计数，
    调用方应把这个计数带到结果里（数据质量记账）。
    """
    if isinstance(ranges_json, str):
        try:
            ranges = json.loads(ranges_json)
        except Exception:  # noqa: BLE001
            return [], 1
    else:
        ranges = ranges_json or []

    out: List[Interval] = []
    bad = 0
    for rg in ranges:
        if not isinstance(rg, dict):
            bad += 1
            continue
        if (rg.get("type") or "").upper() not in ("ECOSYSTEM", "SEMVER"):
            # GIT 范围的 commit 语义无法用于包版本判定，跳过（计入 bad 以便报告）
            bad += 1
            continue
        lo: Optional[Version] = None
        lo_incl = True
        open_ = False
        for ev in rg.get("events") or []:
            if not isinstance(ev, dict):
                bad += 1
                continue
            if "introduced" in ev:
                v = ev["introduced"]
                lo = EPS if str(v).strip() in ("0", "0.0.0", "") else parse_version(v)
                if lo is None:
                    bad += 1
                    lo, lo_incl, open_ = None, True, False
                    continue
                lo_incl, open_ = True, True
            elif "fixed" in ev:
                v = parse_version(ev["fixed"])
                if v is None:
                    bad += 1
                    continue
                if open_:
                    out.append((lo, lo_incl, v, False))
                lo, open_ = None, False
            elif "last_affected" in ev:
                v = parse_version(ev["last_affected"])
                if v is None:
                    bad += 1
                    continue
                if open_:
                    out.append((lo, lo_incl, v, True))
                lo, open_ = None, False
            elif "limit" in ev:
                v = parse_version(ev["limit"])
                if v is not None and open_:
                    out.append((lo, lo_incl, v, False))
                lo, open_ = None, False
            else:
                bad += 1
        if open_:
            out.append((lo, lo_incl, None, False))  # 一直影响到现在
    return out, bad


def in_intervals(v: Optional[Version], intervals: Iterable[Interval]) -> Optional[bool]:
    """None = 判定不了（版本或区间无法解析）。"""
    if v is None:
        return None
    seen = False
    for lo, lo_incl, hi, hi_incl in intervals:
        seen = True
        ok_lo = True if lo is None else (v >= lo if lo_incl else v > lo)
        ok_hi = True if hi is None else (v <= hi if hi_incl else v < hi)
        if ok_lo and ok_hi:
            return True
    return False if seen else None


class VulnIndex:
    """按包聚合的漏洞索引，支持 O(1) 判定"某包某版本是否受影响"。

    记录结构（CSV 一行 = 一个 (advisory, package) 对）：
      advisory_id, package, ranges(json), explicit_versions(;), severity_cvss, published, fixed_versions
    """

    def __init__(self, df):
        import ast

        self.by_pkg: dict[str, list[dict]] = {}
        n_bad_events = 0
        for r in df.itertuples(index=False):
            iv, bad = osv_ranges_to_intervals(getattr(r, "ranges", "[]"))
            n_bad_events += bad
            # 显式版本列表（部分 advisory 只给 versions 不给 ranges）
            vs: List[Version] = []
            if hasattr(r, "explicit_versions") and r.explicit_versions:
                for s in str(r.explicit_versions).split(";"):
                    pv = parse_version(s)
                    if pv is not None:
                        vs.append(pv)
            self.by_pkg.setdefault(r.package, []).append({
                "advisory_id": r.advisory_id,
                "intervals": iv,
                "explicit": vs,
                "cvss": getattr(r, "severity_cvss", None),
                "published": getattr(r, "published", None),
                "summary": getattr(r, "summary", ""),
                "fixed": getattr(r, "fixed_versions", ""),
            })
        self.n_bad_events = n_bad_events

    def packages_with_vulns(self) -> set[str]:
        return set(self.by_pkg)

    def is_vulnerable(self, pkg: str, version: Optional[Version]) -> Optional[bool]:
        """True/False/None(判定不了)。任一条 advisory 命中即为 True。"""
        recs = self.by_pkg.get(pkg)
        if not recs:
            return False
        if version is None:
            return None
        unknown = False
        for rec in recs:
            if rec["explicit"] and version in rec["explicit"]:
                return True
            res = in_intervals(version, rec["intervals"])
            if res is True:
                return True
            if res is None:
                # 区间无法解析时，退化为"该 advisory 列出的 explicit 版本"判定
                if not rec["explicit"]:
                    unknown = True
        return None if unknown else False

    def n_advisories(self, pkg: str) -> int:
        return len(self.by_pkg.get(pkg, []))
