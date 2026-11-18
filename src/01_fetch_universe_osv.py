"""
Step 01 —— 抓取"生态规模"与"漏洞库"两个数据源（快，先跑）。

产出：
  data/raw/top-pypi-packages.min.json        原始下载量榜（逐月，15000 行）
  data/raw/osv_pypi_all.zip                  OSV PyPI 全域 advisory 打包
  data/raw/osv_pypi/*.json                   解压出的单条 advisory
  data/processed/universe.csv                种子集（项目宇宙）
  data/processed/osv_advisories.csv          结构化后的 PyPI advisory 表

为什么要这两个东西？
  * 下载量榜 —— 给我们两样东西：(1) 研究对象边界（前 N 个包）；
    (2) 一个**使用量权重**，用来区分"波及很多包"与"波及很多真实安装"。
  * OSV 漏洞库 —— 提供漏洞的真实版本区间，这是把"结构性级联"收紧成
    "版本约束感知级联"的唯一依据（见 src/31）。
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CFG, PROJECT_ROOT, Log, abspath, canonical_name, ensure_parent, write_json

log = Log("01_fetch_universe_osv")


def http_get(url: str, timeout: int = 120, retries: int = 5) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "csa-final-project/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            wait = 2 ** i
            log(f"  [retry {i+1}/{retries}] {url} -> {e} (sleep {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"GET failed: {url}: {last}")


# ---------------------------------------------------------------- 1. 下载量榜
def fetch_universe() -> pd.DataFrame:
    src = CFG["data_sources"]["top_pypi_packages"]
    raw = abspath(src["raw"])
    log(f"[1/2] 下载量榜 {src['url']}")
    if not raw.exists():
        ensure_parent(raw).write_bytes(http_get(src["url"]))
    doc = json.loads(raw.read_text(encoding="utf-8"))

    rows = doc["rows"]
    df = pd.DataFrame(rows)
    df["project_raw"] = df["project"]
    df["project"] = df["project"].map(canonical_name)
    df = df.drop_duplicates(subset=["project"], keep="first").reset_index(drop=True)
    df["rank"] = df.index + 1
    df = df.rename(columns={"download_count": "downloads"})

    n = int(CFG["graph"]["seed_top_n"])
    universe = df.head(n).copy()
    universe["snapshot"] = doc.get("last_update", "unknown")

    out = ensure_parent(abspath("data/processed/universe.csv"))
    universe[["rank", "project", "project_raw", "downloads", "snapshot"]].to_csv(
        out, index=False, encoding="utf-8"
    )
    log(f"      last_update={doc.get('last_update')}  榜内总行数={len(df)}  取前 {len(universe)}")
    log(f"      下载量合计（前 {len(universe)}）= {universe['downloads'].sum():,}")
    log(f"      -> {out}")
    return universe


# ---------------------------------------------------------------- 2. OSV 漏洞库
def _parse_affected(doc: dict) -> list[dict]:
    """把一条 OSV advisory 展平成 (advisory, package, version-range) 记录。"""
    out = []
    for aff in doc.get("affected", []) or []:
        pkg = aff.get("package", {}) or {}
        if (pkg.get("ecosystem") or "").lower() != "pypi":
            continue
        name = canonical_name(pkg.get("name") or "")
        if not name:
            continue
        ranges = []
        for rg in aff.get("ranges", []) or []:
            ranges.append({
                "type": rg.get("type"),
                "events": rg.get("events", []),
            })
        out.append({
            "advisory_id": doc.get("id"),
            "aliases": ";".join(doc.get("aliases", []) or []),
            "summary": _scrub_webhooks((doc.get("summary") or "").replace("\n", " "))[:300],
            "published": doc.get("published"),
            "modified": doc.get("modified"),
            "severity_cvss": _cvss_score(doc),
            "package": name,
            "ranges": json.dumps(ranges, ensure_ascii=False),
            "versions_count": len(aff.get("versions", []) or []),
            "explicit_versions": ";".join(aff.get("versions", []) or []),
            "database_specific": _scrub_webhooks(
                json.dumps(doc.get("database_specific", {}), ensure_ascii=False))[:500],
            "fixed_versions": ";".join(_fixed_versions(doc)),
        })
    return out


# ---------------------------------------------------------------- 凭据清理
# 为什么需要这一步（不是为了好看，是为了不帮倒忙）
# ----------------------------------------------------------------
# OSV 的 MAL-* 恶意包公告里会原样记录**攻击者的外泄端点**——本项目抽到的
# PyPI 子集里就有 1 个 Slack webhook 与 87 个 Discord webhook，
# 全部出现在 `database_specific` 字段（例如 MAL-2024-11627 / luna-paysure 会把
# 受害者数据 POST 到那个地址）。它们是攻击者仍然在用的基础设施，不是我们的凭据。
#
# 我们把这些 URL 抹成 .../REDACTED，理由有三条：
#   1) 转载活的攻击基础设施没有正当理由；OSV 原文仍然公开可取（见 data/raw 的原始 zip）；
#   2) GitHub 的 push protection 会直接拦下含 webhook 的推送，留着它就推不上仓库；
#   3) 对分析零损失——本项目的所有计算只用 advisory_id / package / ranges /
#      versions / cvss / fixed_versions，从不读 summary 与 database_specific 的正文。
#
# 注意清理只作用于**派生表**；data/raw/osv_pypi_all.zip 保持原样以保证溯源完整。
_WEBHOOK_RE = re.compile(
    r"https?://(?:"
    r"hooks\.slack\.com/services/[A-Za-z0-9/_-]+"
    # Discord 的 webhook 形如 /api/webhooks/{id}/{token}，但数据里也有只有 id、
    # 或 id 后带一个空尾斜杠的写法，所以 token 段必须允许缺失。
    r"|(?:ptb\.|canary\.)?(?:discord|discordapp)\.com/api/webhooks/[0-9]+(?:/[A-Za-z0-9_-]*)?"
    r")"
)


def _scrub_webhooks(text: str) -> str:
    """把文本里的 Slack / Discord webhook 端点替换为 REDACTED。"""
    return _WEBHOOK_RE.sub("https://REDACTED.invalid/webhook", text or "")


def _cvss_score(doc: dict) -> float | None:
    for s in doc.get("severity", []) or []:
        if (s.get("type") or "").upper() == "CVSS_V3":
            sc = _cvss3_base(s.get("score", ""))
            if sc is not None:
                return sc
    for s in doc.get("severity", []) or []:
        if (s.get("type") or "").upper() == "CVSS_V4":
            return None  # 只保留 CVSSv3，避免混合量纲
    db = doc.get("database_specific", {}) or {}
    sev = (db.get("severity") or "").upper()
    return {"CRITICAL": 9.5, "HIGH": 8.0, "MODERATE": 5.5, "MEDIUM": 5.5, "LOW": 2.5}.get(sev)


def _cvss3_base(vector: str) -> float | None:
    """按 CVSS v3.1 规范计算 base score（不依赖外部库）。"""
    m = dict(re.findall(r"([A-Z]{1,2}):([A-Z])", vector or ""))
    need = {"AV", "AC", "PR", "UI", "S", "C", "I", "A"}
    if not need.issubset(m):
        return None
    AV = {"N": .85, "A": .62, "L": .55, "P": .2}[m["AV"]]
    AC = {"L": .77, "H": .44}[m["AC"]]
    S_changed = m["S"] == "C"
    PR = ({"N": .85, "L": .68, "H": .50} if S_changed else {"N": .85, "L": .62, "H": .27})[m["PR"]]
    UI = {"N": .85, "R": .62}[m["UI"]]
    w = {"N": 0.0, "L": .22, "H": .56}
    C, I, A = w[m["C"]], w[m["I"]], w[m["A"]]
    iss = 1 - (1 - C) * (1 - I) * (1 - A)
    if S_changed:
        imp = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15
    else:
        imp = 6.42 * iss
    if imp <= 0:
        return 0.0
    expl = 8.22 * AV * AC * PR * UI
    raw = min((1.08 if S_changed else 1.0) * (imp + expl), 10.0)
    return math.ceil(raw * 10) / 10


def _fixed_versions(doc: dict) -> list[str]:
    out = []
    for aff in doc.get("affected", []) or []:
        if (aff.get("package", {}) or {}).get("ecosystem", "").lower() != "pypi":
            continue
        for rg in aff.get("ranges", []) or []:
            for ev in rg.get("events", []) or []:
                if "fixed" in ev:
                    out.append(str(ev["fixed"]))
    return sorted(set(out))


def fetch_osv() -> pd.DataFrame:
    src = CFG["data_sources"]["osv"]
    zpath = abspath(src["raw_zip"])
    log(f"[2/2] OSV PyPI 漏洞库 {src['url']}")
    if not zpath.exists():
        ensure_parent(zpath).write_bytes(http_get(src["url"], timeout=300))
    log(f"      zip 大小 {zpath.stat().st_size/1e6:.1f} MB")

    exdir = abspath(src["extract_dir"])
    exdir.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(zpath)
    names = [n for n in zf.namelist() if n.endswith(".json")]
    log(f"      advisory 文件数 = {len(names)}")

    rows: list[dict] = []
    bad = 0
    for n in names:
        try:
            doc = json.loads(zf.read(n).decode("utf-8"))
        except Exception:  # noqa: BLE001
            bad += 1
            continue
        # 只保留"针对 PyPI 包本身"的 advisory
        if (doc.get("affected") or [{}])[0].get("package", {}).get("ecosystem", "").lower() != "pypi":
            pass
        rows.extend(_parse_affected(doc))

    df = pd.DataFrame(rows)
    out = ensure_parent(abspath("data/processed/osv_advisories.csv"))
    df.to_csv(out, index=False, encoding="utf-8")
    log(f"      解析失败 {bad} 条；PyPI 受影响包记录 {len(df)} 行；涉及 {df['package'].nunique()} 个包")
    log(f"      advisory 唯一 ID 数 = {df['advisory_id'].nunique()}")
    log(f"      -> {out}")
    return df


if __name__ == "__main__":
    t0 = time.time()
    uni = fetch_universe()
    osv = fetch_osv()
    log(f"完成，用时 {time.time()-t0:.1f}s")
    log.close()
