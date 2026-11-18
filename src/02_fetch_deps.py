"""
Step 02 —— 抓取"已解析依赖闭包"（deps.dev v3 API）。

为什么用 deps.dev 而不是 PyPI 的 requires_dist？
----------------------------------------------------------------
PyPI 元数据里写的是 PEP 508 的**约束表达式**（如 `urllib3>=1.21.1,<3`），
不是解析结果。直接用它建图会遇到三个问题：
  1) 约束区间重叠时，A 声明 -> 一家实现（extra、环境 marker）可能不成立；
  2) 同一生态里不同包解析出的版本互相冲突（版本求解问题）；
  3) 很多包的依赖是"有条件"的（`; extra == 'test'`、`; python_version<'3.8'`）。

deps.dev（Google osv-scanner 团队维护）返回的是**一个具体版本**的
已解析依赖闭包（version resolution 已完成），每个依赖节点带 DIRECT/INDIRECT
关系与解析后的确切版本号。它同时提供了我们需要的"版本"信息，
使"版本约束感知的级联"（src/31）成为可计算的问题。

本脚本对每个种子包：
  1. GET /systems/pypi/packages/{name}                -> 找出 isDefault 的默认版本
  2. GET /systems/pypi/packages/{name}/versions/{v}:dependencies -> 已解析闭包

产出（逐行 JSON，可断点续跑）：
  data/raw/depsdev_deps.jsonl
  data/raw/depsdev_failed.jsonl
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import abspath, canonical_name, CFG, ensure_parent, Log, read_universe

log = Log("02_fetch_deps")
DS = CFG["data_sources"]["depsdev"]
BASE = DS["base"]
OUT = abspath(DS["raw_jsonl"])
FAIL = abspath(DS["failed_jsonl"])
ensure_parent(OUT)
ensure_parent(FAIL)

# ------------------------------------------------------------------ 限速器
class RateLimiter:
    def __init__(self, qps: float):
        self.min_interval = 1.0 / qps
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now < self.next_t:
                time.sleep(self.next_t - now)
                now = time.monotonic()
            self.next_t = now + self.min_interval


RL = RateLimiter(DS["max_qps"])


def _get(url: str) -> dict:
    """带重试的 GET；404 视为"该包不存在"，直接返回 None。"""
    last = None
    for attempt in range(DS["retries"]):
        RL.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "csa-final-project/1.0"})
            with urllib.request.urlopen(req, timeout=DS["timeout_s"]) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (429, 500, 502, 503, 504):
                last = e
                time.sleep(2 ** attempt)
                continue
            last = e
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(str(last))


def fetch_one(name: str) -> dict:
    """返回一条完整记录；网络层重试耗尽才抛异常。

    数据质量记账（重要）：
      deps_status = "ok"            -> deps.dev 给出了已解析闭包
      deps_status = "unresolved"    -> deps.dev 有包但该版本无解析结果(404)，
                                       回退到 PyPI 的 requires_dist（只含直接依赖声明）
      deps_status = "missing"       -> 连包都没有
    我们**不**把"解析失败"当成"没有依赖"，这是本项目最关键的数据清洗口径之一。
    """
    # --- 1) 默认版本
    info = _get(f"{BASE}/packages/{name}")
    if info is None:
        return {"name": name, "status": "not_found", "deps_status": "missing", "nodes": []}
    versions = info.get("versions") or []
    default = next((v for v in versions if v.get("isDefault")), None)
    if default is None and versions:
        default = versions[-1]
    if default is None:
        return {"name": name, "status": "no_versions", "deps_status": "missing", "nodes": []}
    ver = default["versionKey"]["version"]

    # --- 2) 已解析依赖闭包
    dep = _get(f"{BASE}/packages/{name}/versions/{ver}:dependencies")
    nodes: list[dict] = []
    edges = None
    deps_status = "ok"
    if dep is None:
        deps_status = "unresolved"
        nodes, src_note = _fallback_pypi_requires(name, ver)
    else:
        src_note = "depsdev"
        for nd in dep.get("nodes", []):
            vk = nd.get("versionKey", {}) or {}
            nodes.append({
                "name": canonical_name(vk.get("name") or ""),
                "version": vk.get("version"),
                "relation": nd.get("relation"),
                "bundled": nd.get("bundled", False),
                "errors": nd.get("errors") or [],
            })
        edges = dep.get("edges")

    return {
        "name": name,
        "status": "ok",
        "deps_status": deps_status,
        "source": src_note,
        "resolved_version": ver,
        "latest_version": (info.get("versions") or [{}])[-1].get("versionKey", {}).get("version"),
        "published_at": default.get("publishedAt"),
        "n_dep_nodes": len(nodes),
        "n_direct": sum(1 for n in nodes if n["relation"] == "DIRECT"),
        "n_indirect": sum(1 for n in nodes if n["relation"] == "INDIRECT"),
        "nodes": nodes,
        "edges": edges,
    }


def _fallback_pypi_requires(name: str, ver: str) -> tuple[list[dict], str]:
    """回退数据源：PyPI 的 Requires-Dist（PEP 508 直接依赖声明）。

    注意语义差异：这里拿到的是**直接依赖的约束表达式**，不是解析结果。
    我们在 relation 里标 "DIRECT_DECLARED" 以便后续实验区分。
    只为提高覆盖率，不改变"已解析优先"的原则。
    """
    try:
        doc = _get_json(f"https://pypi.org/pypi/{name}/{ver}/json")
    except Exception:  # noqa: BLE001
        return [], "none"
    if not doc:
        return [], "none"
    rd = (doc.get("info") or {}).get("requires_dist") or []
    out, seen = [], set()
    for spec in rd:
        try:
            from packaging.requirements import Requirement
            req = Requirement(spec)
        except Exception:  # noqa: BLE001
            continue
        # 只保留"无条件"依赖：剔除 extra 才启用的可选依赖
        if req.marker is not None and "extra" in str(req.marker):
            continue
        nm = canonical_name(req.name)
        if nm in seen:
            continue
        seen.add(nm)
        out.append({"name": nm, "version": None, "relation": "DIRECT_DECLARED",
                    "bundled": False, "errors": [], "specifier": str(req.specifier)})
    return out, "pypi_requires_dist"


def _get_json(url: str) -> dict | None:
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "csa-final-project/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 * (attempt + 1))
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (attempt + 1))
    return None


def load_done() -> set[str]:
    done = set()
    for p in (OUT,):
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["name"])
                    except Exception:  # noqa: BLE001
                        continue
    return done


def main() -> None:
    uni = read_universe()
    names = uni["project"].tolist()
    done = load_done()
    todo = [n for n in names if n not in done]
    log(f"种子包 {len(names)}，已完成 {len(done)}，待抓 {len(todo)}")

    if not todo:
        log("全部完成，退出")
        return

    fout = open(OUT, "a", encoding="utf-8")
    ffail = open(FAIL, "a", encoding="utf-8")
    t0 = time.time()
    cnt = ok = 0
    lock = threading.Lock()

    def work(nm: str) -> None:
        nonlocal cnt, ok
        try:
            rec = fetch_one(nm)
        except Exception as e:  # noqa: BLE001
            rec = {"name": nm, "status": "error", "error": str(e)}
        with lock:
            if rec["status"] == "ok":
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ok += 1
            else:
                ffail.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cnt += 1
            if cnt % 250 == 0:
                el = time.time() - t0
                rate = cnt / el
                eta = (len(todo) - cnt) / rate if rate else 0
                log(f"  进度 {cnt}/{len(todo)}  ok={ok}  {rate:.1f} 包/s  ETA {eta/60:.1f} min")
                fout.flush()
                ffail.flush()

    try:
        with ThreadPoolExecutor(max_workers=DS["concurrency"]) as ex:
            futs = [ex.submit(work, n) for n in todo]
            for _ in as_completed(futs):
                pass
    finally:
        fout.close()
        ffail.close()

    log(f"完成：成功 {ok}，失败 {cnt-ok}，用时 {(time.time()-t0)/60:.1f} min")
    log(f"-> {OUT}")
    log(f"-> {FAIL}")
    log.close()


if __name__ == "__main__":
    main()
