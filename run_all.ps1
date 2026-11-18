<#
.SYNOPSIS
    复杂系统分析 期末项目 —— 一键复现（Windows PowerShell 5.1 / PowerShell 7 均可）

.DESCRIPTION
    顺序执行：抓取 → 建图 → 实验一..四 → 对照 → 验证 → 敏感性 → 图表 → PPT
    所有参数来自 config/config.yaml（唯一参数真源）。

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File .\run_all.ps1 -SkipFetch
#>
param(
    [switch]$SkipFetch
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
# PYTHONHASHSEED=0：字符串哈希随机化会让 set 的迭代顺序在不同进程间不同，
# 而零模型的边随机化从一个 set 起步 —— 不固定它，同一 seed 在不同进程里会得到
# 略有差别的结果（本项目自查时发现过约 0.02% 的漂移）。固定后逐位可复现。
$env:PYTHONHASHSEED = "0"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

function Step($n, $title, $file) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor Cyan
    Write-Host ("[{0}] {1}" -f $n, $title) -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor Cyan
    $t0 = Get-Date
    python -X utf8 $file
    if ($LASTEXITCODE -ne 0) { throw ("步骤 {0} 失败：{1}" -f $n, $file) }
    Write-Host ("    用时 {0:N1}s" -f ((Get-Date) - $t0).TotalSeconds) -ForegroundColor Green
}

if (-not $SkipFetch) {
    Step "01" "抓取下载量榜 + OSV 漏洞库"                    "src/01_fetch_universe_osv.py"
    Step "02" "抓取已解析依赖闭包（约 17 分钟）"              "src/02_fetch_deps.py"
}
Step "10" "构建有向依赖网络 + 数据质量记账"                  "src/10_build_graph.py"
Step "20" "实验一：结构描述"                                "src/20_exp_structure.py"
Step "21" "实验二：级联影响面与瓶颈分数"                     "src/21_exp_cascade.py"
Step "22" "实验三：三级口径分解"                             "src/22_exp_vuln_tiers.py"
Step "23" "实验四：韧性情景（AND/OR、修复/移除）"            "src/23_exp_resilience.py"
Step "24" "对照实验：保度序列零模型"                         "src/24_exp_nullmodel.py"
Step "25" "验证实验：预测力与时间外推"                       "src/25_exp_validation.py"
Step "26" "敏感性：排序更新协议（固定 vs 自适应）"            "src/26_exp_adaptive.py"
Step "30" "生成图表"                                        "src/30_figures.py"
Step "40" "生成答辩 PPT"                                    "src/40_make_pptx.py"

Write-Host ""
Write-Host "全部完成。结果见 results/tables 与 results/figures，报告见 report/。" -ForegroundColor Green
