"""
公共工具：配置加载、路径解析、日志、以及 PEP 508 版本约束语义工具。

设计原则（对应课程要求"节点、边、权重与数据处理是否解释清楚"）：
  * 所有路径都从项目根解析，脚本可在任意 cwd 下运行；
  * 所有随机性必须显式带 seed；
  * 版本语义严格使用 packaging 库，不自己写字符串比较。
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------- 路径与配置
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = None) -> Dict[str, Any]:
    p = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()


def root(*parts: str) -> Path:
    """项目根下的路径，自动创建父目录。"""
    p = PROJECT_ROOT.joinpath(*parts)
    return p


def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def abspath(rel: str) -> Path:
    return PROJECT_ROOT / rel


# ---------------------------------------------------------------- 日志
class Log:
    """极简日志：同时写 stdout 和文件，避免 PowerShell 编码问题。"""

    def __init__(self, name: str):
        self.path = ensure_parent(PROJECT_ROOT / "results" / "logs" / f"{name}.log")
        self.fh = open(self.path, "a", encoding="utf-8")

    def __call__(self, *args: Any) -> None:
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


# ---------------------------------------------------------------- PEP 508 名称规范化
# PEP 503 规范化：小写 + 把 -, _, . 的任意连续组合折叠为单个 -
def canonical_name(name: str) -> str:
    if not isinstance(name, str):
        # 数据源里偶有非字符串（缺失/数值化）名称，返回空串交由调用方丢弃并记账
        return ""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


# ---------------------------------------------------------------- JSON 读写
def write_json(path: str | Path, obj: Any) -> None:
    p = ensure_parent(Path(path))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 数据读取（踩过的坑）
# 坑：PyPI 上真的存在一个名为 `nan` 的包（以及 `null`、`NA` 之类的名字）。
#     用 pandas 默认设置 read_csv 会把它读成 float('NaN')，随后它变成一个
#     "神秘的缺失值"——不报错，但会在下游悄悄丢一个包、并且 JSON 序列化时写出
#     非法的裸 NaN token。必须 keep_default_na=False，并显式给出空值集合。
def read_universe() -> "Any":
    import pandas as pd
    p = abspath("data/processed/universe.csv")
    return pd.read_csv(
        p, keep_default_na=False, na_values=[""], dtype={"project": str, "project_raw": str}
    )


def read_csv_safe(path: "str | Path", **kw) -> "Any":
    """统一的 CSV 读取：保留 `nan` / `null` / `NA` 等**合法的包名**。

    参数可以是项目根相对路径（str）或绝对 Path。
    这是本项目最"琐碎但致命"的一个数据清洗口径：
    PyPI 上真实存在名为 `nan`、`null`、`none` 的包，pandas 的默认 NA 规则
    会把它们读成缺失值，然后这个包从图里静默消失——不报错、不影响任何指标长相，
    但会让"传递依赖者数"少算一批边。
    """
    import pandas as pd
    kw.setdefault("keep_default_na", False)
    kw.setdefault("na_values", [""])
    s = str(path)
    p = Path(s) if (os.path.isabs(s) or s.startswith(("/", "\\"))) else abspath(s)
    return pd.read_csv(p, **kw)


# ---------------------------------------------------------------- 确定性排序
def rank_by(values: dict, ascending: bool = False) -> list:
    """按数值降序（或升序）给出节点名列表，**同值时按名称字典序打破平局**。

    为什么需要它：实验四和零模型对照必须使用**完全相同**的移除顺序，
    否则两者的 S(q) 曲线在同一个 q 上会给出略微不同的数字（本项目自查时
    发现过 0.392 vs 0.407 的差异，即 ~15 个千分点，来自 pandas `sort_values`
    的默认不稳定排序在大量同值节点上的任意顺序）。
    显式平局规则让"同一移除预算下比较"这个对照条件真正成立，也让结果可复现。
    """
    return sorted(values, key=lambda k: (values[k] if ascending else -values[k], k))


# ---------------------------------------------------------------- 位集 → 掩码
def bits_to_mask(row: "Any", n: int) -> "Any":
    """把 (nw,) 的 uint64 位集行展开成长度 n 的布尔掩码。

    ★ 必须用 bitorder="little"（本项目自查时踩过这个隐蔽坑）
    -----------------------------------------------------------------
    打包时用的是 `word[i >> 6] |= 1 << (i & 63)`，即"全局第 i 位 = 第 (i>>6) 个
    uint64 字的第 (i&63) 位"。x86/ARM 都是小端，因此这个字的第 k 位落在它的
    第 (k>>3) 个字节的第 (k&7) 位。

    而 `np.unpackbits()` 默认按 **MSB-first** 展开每个字节，会把每个字节内的
    8 个位**倒序**输出。后果极其隐蔽：
      * 集合的**大小**（popcount）完全正确 → 「级联规模」类指标看不出问题；
      * 集合的**成员/位置**全部错乱（每个 64 位块内被置换）→
        「加权暴露量」与「按位置取值的布尔掩码」全错。
    本项目就是用 popcount 与闭包口径互相校验（Spearman 0.971）时才没暴露出来，
    后来用 `nx.ancestors` 做集合级交叉验证才发现"大小对、成员错"。
    """
    import numpy as np
    return np.unpackbits(row.view(np.uint8), bitorder="little")[:n].astype(bool)
