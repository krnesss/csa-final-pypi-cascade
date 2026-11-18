# data/processed —— 派生数据说明

这个目录里的文件全部由 `src/01_fetch_universe_osv.py` 与 `src/10_build_graph.py`
从 `data/raw/` 的原始文件**确定性生成**。它们被入库存档，是因为本项目的研究结论
依赖一个**固定时间切片**，而这些文件就是"这个切片"的证据：

| 文件 | 内容 | 作用 |
|---|---|---|
| `universe.csv` | 下载量榜前 15,000 名 + 榜单的 `last_update` | 定义研究的**对象边界**与**时间切片** |
| `universe_coverage.csv` | 每个种子包被解析到的**确切版本号** + 解析状态 | 让切片可逐包重建；记录 525 个回退包 |
| `closures.parquet` | 每个种子的已解析闭包（seed, dep, **dep_version**, relation） | **版本感知级联判定**的原料（实验三的核心） |
| `nodes.csv` / `edges.csv` | 有向依赖网络的节点表与直接依赖边表 | 15,718 节点 / 47,273 边 |
| `node_metrics.parquet` | 每个包的 8 个瓶颈指标 + 广度/深度/加权暴露 | 实验二、四的输入 |
| `degrees.parquet` | 入度 / 出度 / 是否种子 / 下载量 | 实验一的输入 |
| `reach_bits.npy` | 15,718 × 246 的 `uint64` 反向可达位集矩阵 | 实验四反复复用的加速缓存 |
| `reach_nodes.json` | 位集矩阵的行顺序 | 位集 ↔ 包名的映射 |
| `graph_stats.json` | 网络规模与**数据质量记账**（环、孤立点、覆盖率…） | 审计用 |
| `osv_advisories.csv` | OSV 公告展平表（区间、显式版本、CVSS、修复版本） | 实验三的输入 |

---

## 一处刻意的数据修订：webhook 端点已脱敏

`osv_advisories.csv` 的 `summary` / `database_specific` 字段里，OSV 会原样记录
**恶意包（`MAL-*`）的攻击者外泄端点**。本项目的 PyPI 子集抽到 **1 个 Slack webhook
与 87 个 Discord webhook**（例如 `MAL-2024-11627` / 包 `luna-paysure` 会把受害者数据
POST 到那个地址）。它们是攻击者仍在使用的活体基础设施，不是本项目的凭据。

`src/01_fetch_universe_osv.py` 的 `_scrub_webhooks()` 把这类 URL 替换为
`https://REDACTED.invalid/webhook`（共 231 处）。三条理由：

1. **转载活的攻击基础设施没有正当理由**；OSV 原文仍然公开可取（见下方原始文件 URL）。
2. **GitHub push protection 会直接拦下含 webhook 的推送**，留着就推不上仓库。
3. **对分析零损失**：本项目所有计算只用 `advisory_id / package / ranges /
   explicit_versions / severity_cvss / fixed_versions`，从不读 `summary` 与
   `database_specific` 的正文。

**可验证性**：脱敏前后，上述"被使用的列"的 `hash_pandas_object` 指纹**完全一致**
（行数 31,957、涉及 13,695 个包均不变）。脱敏只作用于**派生表**；
`data/raw/osv_pypi_all.zip` 保持原样，溯源完整。

---

## 原始文件（`data/raw/`）的来源

全部公开、无需密钥，可用 `run_all.ps1` 重新抓取：

| 文件 | URL |
|---|---|
| `top-pypi-packages.min.json` | `https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json`（镜像 pypistats 的 ClickHouse 统计） |
| `depsdev_deps.jsonl` | `https://api.deps.dev/v3/systems/pypi/...`（Google osv-scanner 团队，返回**已解析**依赖闭包） |
| `depsdev_failed.jsonl` | 同上，抓取失败/包不存在的记录（7 个包，诚实记账） |
| `osv_pypi_all.zip` | `https://osv-vulnerabilities.storage.googleapis.com/PyPI/all.zip` |

> 注：`osv_pypi_all.zip` 是 OSV 的**公开数据集原样镜像**，因此它内部仍然含有上述
> webhook 字符串。我们不改动原始数据以保持溯源完整；只有派生表做了脱敏。
