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

用法：
  python src/02_fetch_deps.py                 # 只抓还没抓过的种子包（断点续跑）
  python src/02_fetch_deps.py --refetch-weak  # 重抓"回退且抓取失败"的弱记录（评审 A-01）
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
    # 评审意见 A-01：把"拿到解析对象"与"解析确实无错且完整"分开记账。
    # 原来只要 API 返回了对象就记 ok，并把 ok 当成"闭包完整"，这是过度声称。
    # 这里额外保存图级 error 原字段与节点级错误计数，供 10_build_graph.py 做严格判定。
    graph_error = None
    n_node_errors = 0
    if dep is None:
        deps_status = "unresolved"
        nodes, src_note, fb_status = _fallback_pypi_requires(name, ver)
    else:
        src_note = "depsdev"
        fb_status = None
        # deps.dev 的响应里可能有图级错误字段（不同版本字段名不一，逐个探测并保留）
        for key in ("error", "errors"):
            if dep.get(key):
                graph_error = dep.get(key)
                break
        for nd in dep.get("nodes", []):
            vk = nd.get("versionKey", {}) or {}
            nd_errors = nd.get("errors") or []
            if nd_errors:
                n_node_errors += 1
            nodes.append({
                "name": canonical_name(vk.get("name") or ""),
                "version": vk.get("version"),
                "relation": nd.get("relation"),
                "bundled": nd.get("bundled", False),
                "errors": nd_errors,
            })
        edges = dep.get("edges")

    return {
        "name": name,
        "status": "ok",
        "deps_status": deps_status,
        "source": src_note,
        "fallback_status": fb_status,
        # 严格完整性判据（A-01）：只有"拿到了图且图级无错且无节点级错误"才算严格完整
        "closure_complete_strict": bool(
            deps_status == "ok" and not graph_error and n_node_errors == 0)
            if deps_status == "ok" else bool(fb_status == "ok"),
        "graph_error": graph_error,
        "n_node_errors": n_node_errors,
        "resolved_version": ver,
        "latest_version": (info.get("versions") or [{}])[-1].get("versionKey", {}).get("version"),
        "published_at": default.get("publishedAt"),
        "n_dep_nodes": len(nodes),
        "n_direct": sum(1 for n in nodes if n["relation"] == "DIRECT"),
        "n_indirect": sum(1 for n in nodes if n["relation"] == "INDIRECT"),
        "nodes": nodes,
        "edges": edges,
    }


def _marker_env() -> dict:
    """基准环境（评审意见 A-03）：marker 必须在**显式固定**的环境里求值。

    原来只判断 marker 字符串里有没有 "extra"，对 `python_version < "3.11"`、
    `sys_platform == "win32"` 这类条件完全不求值，于是可能把与假定环境不符的
    边混进 DIRECT_DECLARED。现在用 config 里固定的基准环境做 evaluate()：
      - `extra` 固定为空串 ⇒ 即"默认安装、不启用任何 extra"；
      - python_version / sys_platform 等全部钉死，结果与跑脚本的机器无关。
    """
    env = dict(CFG["experiments"]["fallback_marker_environment"])
    # packaging 要求这些键存在，缺失时用它的默认值补齐（其余键保持我们固定的值）
    from packaging.markers import default_environment
    base = default_environment()
    base.update({k: str(v) for k, v in env.items()})
    return base


def _fallback_pypi_requires(name: str, ver: str) -> tuple[list[dict], str, str]:
    """回退数据源：PyPI 的 Requires-Dist（PEP 508 直接依赖声明）。

    注意语义差异：这里拿到的是**直接依赖的约束表达式**，不是解析结果。
    我们在 relation 里标 "DIRECT_DECLARED" 以便后续实验区分。
    只为提高覆盖率，不改变"已解析优先"的原则。

    评审意见 A-03：marker 用 `req.marker.evaluate(environment=基准环境)` 真值判定，
    而不是看字符串里有没有 "extra"。

    评审意见 A-01：返回第三个值 `fallback_status`，把
      "ok"                —— 拿到了 requires_dist 并按 marker 过滤完
      "no_requires_dist"  —— PyPI 明确没有声明任何依赖（真的是空）
      "fetch_failed"      —— PyPI 请求失败（**不能**当成"没有依赖"，必须记账）
    区分开。原来把三者都变成"空列表"，等于把抓取失败静默当成"无依赖"。
    """
    doc = _get_json(f"https://pypi.org/pypi/{name}/{ver}/json")
    if doc is None:
        return [], "none", "fetch_failed"
    info = doc.get("info") or {}
    rd = info.get("requires_dist") or []
    if not rd:
        return [], "pypi_requires_dist", "no_requires_dist"
    env = _marker_env()
    out, seen = [], set()
    for spec in rd:
        try:
            from packaging.requirements import Requirement
            req = Requirement(spec)
        except Exception:  # noqa: BLE001
            continue
        if req.marker is not None:
            try:
                # extra 固定为 ""（默认安装）；其余 marker 在固定环境里求值
                if not req.marker.evaluate(environment=env):
                    continue
            except Exception:  # noqa: BLE001
                # 表达式无法求值 → 丢弃（属于"条件不可判定"，由 A-01 统计）
                continue
        nm = canonical_name(req.name)
        if nm in seen:
            continue
        seen.add(nm)
        out.append({"name": nm, "version": None, "relation": "DIRECT_DECLARED",
                    "bundled": False, "errors": [], "specifier": str(req.specifier),
                    "marker": str(req.marker) if req.marker else ""})
    return out, "pypi_requires_dist", "ok"


def _get_json(url: str) -> dict | None:
    # 评审意见 A-01 相关：并发打 pypi.org 容易被限流，进而静默返回 None，
    # 让"抓取失败"看起来像"没有依赖"。这里显式走同一个限速器。
    for attempt in range(4):
        RL.wait()
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


def load_records() -> list[dict]:
    """读回逐行 JSON；同名记录以**最后一条**为准（重抓后追加是常态）。"""
    by_name: dict[str, dict] = {}
    order: list[str] = []
    if OUT.exists():
        with open(OUT, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                nm = rec.get("name")
                if nm is None:
                    continue
                if nm not in by_name:
                    order.append(nm)
                by_name[nm] = rec
    return [by_name[n] for n in order]


# 评审意见 A-01：什么叫"这条记录真的可以算完成"。
# 老口径只看名字出现过没有 —— 于是"deps.dev 没解析 + PyPI 抓取失败 + 两者都被
# 记成空依赖列表"的记录也会被当成已完成，永远不再重试，错误就被固化了。
#   已解析（deps_status=="ok"）          -> 完成，永不重抓
#   回退且 status=="ok"                  -> 完成（依赖真的拿到了）
#   回退且 status=="no_requires_dist"    -> 完成（PyPI 明确没有依赖声明）
#   回退且 status 缺失/=="fetch_failed"  -> **弱记录**，可被 --refetch-weak 重抓
def is_weak(rec: dict) -> bool:
    if rec.get("deps_status") == "ok":
        return False
    return rec.get("fallback_status") not in ("ok", "no_requires_dist")


def main() -> None:
    refetch_weak = "--refetch-weak" in sys.argv[1:]
    uni = read_universe()
    names = uni["project"].tolist()
    keep = load_records()
    if refetch_weak:
        strong = [r for r in keep if not is_weak(r)]
        weak = [r for r in keep if is_weak(r)]
        if weak:
            tmp = OUT.with_suffix(OUT.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for r in strong:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, OUT)
        log(f"--refetch-weak：保留 {len(strong)} 条强记录，重抓 {len(weak)} 条弱记录")
    done = {r["name"] for r in keep if not (refetch_weak and is_weak(r))}
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
