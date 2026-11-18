"""
Step 40 —— 生成答辩用 PPT（.pptx，16:9，可直接用 PowerPoint / WPS 打开）

设计约束（针对 20 分钟答辩）：
  * 每页只讲一个论点，标题就是论点而不是名词；
  * 图优先，文字是图的注解；
  * 关键数字用大字号强调；
  * 每页右下角有对应结果文件路径，方便评委追问时说"这个数在哪个表里"。

产出：report/答辩PPT.pptx
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CFG, PROJECT_ROOT, Log, abspath, read_csv_safe

log = Log("40_make_pptx")
FIG = abspath(CFG["figures"]["outdir"])
TBL = abspath(CFG["tables"]["outdir"])
OUT = PROJECT_ROOT / "report" / "答辩PPT.pptx"

# ---------------- 配色 ----------------
INK = RGBColor(0x1B, 0x25, 0x2B)
ACC = RGBColor(0xC0, 0x39, 0x2B)   # 强调红
ACC2 = RGBColor(0x16, 0x8A, 0x7E)  # 强调绿
GRAY = RGBColor(0x6B, 0x77, 0x80)
LIGHT = RGBColor(0xF4, 0xF6, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = prs.slide_width, prs.slide_height


def _tb(slide, x, y, w, h, text, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT,
        space_after=6, line=1.15, font="Microsoft YaHei"):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    lines = text.split("\n") if isinstance(text, str) else text
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.line_spacing = line
        r = p.add_run()
        r.text = ln
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = font
    return box


def _rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    s.shadow.inherit = False
    return s


def new_slide(title=None, kicker=None, source=None, bg=WHITE):
    s = prs.slides.add_slide(BLANK)
    _rect(s, 0, 0, W, H, bg)
    if title:
        _rect(s, Inches(0.6), Inches(0.52), Inches(0.09), Inches(0.62), ACC)
        _tb(s, Inches(0.85), Inches(0.40), Inches(11.6), Inches(0.9), title,
            size=27, bold=True, color=INK)
    if kicker:
        _tb(s, Inches(0.88), Inches(1.14), Inches(11.8), Inches(0.45), kicker,
            size=13.5, color=GRAY)
    if source:
        _tb(s, Inches(0.6), H - Inches(0.52), Inches(12.1), Inches(0.35),
            source, size=9.5, color=GRAY)
    return s


def pic(s, name, x, y, w=None, h=None):
    p = FIG / f"{name}.png"
    if not p.exists():
        log(f"  [缺图] {name}")
        return
    ph = s.shapes.add_picture(str(p), x, y, width=w, height=h)
    return ph


def fit_pic(s, name, x, y, max_w, max_h):
    """等比缩放到给定框内并居中。"""
    from PIL import Image
    p = FIG / f"{name}.png"
    if not p.exists():
        log(f"  [缺图] {name}")
        return
    iw, ih = Image.open(p).size
    r = min(max_w / iw, max_h / ih)
    w, h = int(iw * r), int(ih * r)
    return s.shapes.add_picture(str(p), int(x + (max_w - w) / 2), int(y + (max_h - h) / 2),
                               width=w, height=h)


def kv_block(s, x, y, w, items, big=34, small=12):
    """一排大数字 + 小标签。"""
    n = len(items)
    cw = int(w / n)
    for i, (val, lab) in enumerate(items):
        cx = x + i * cw
        _rect(s, cx, y, cw - Inches(0.18), Inches(1.32), LIGHT)
        _tb(s, cx + Inches(0.12), y + Inches(0.10), cw - Inches(0.4), Inches(0.7),
            val, size=big, bold=True, color=ACC, align=PP_ALIGN.CENTER)
        _tb(s, cx + Inches(0.12), y + Inches(0.78), cw - Inches(0.4), Inches(0.5),
            lab, size=small, color=GRAY, align=PP_ALIGN.CENTER)


def bullets(s, x, y, w, items, size=16, gap=0.62):
    for i, (head, body) in enumerate(items):
        yy = y + Inches(gap * i)
        _rect(s, x, yy + Inches(0.10), Inches(0.07), Inches(0.32), ACC2)
        _tb(s, x + Inches(0.20), yy, w, Inches(0.4), head, size=size, bold=True)
        if body:
            _tb(s, x + Inches(0.20), yy + Inches(0.34), w, Inches(0.5), body,
                size=size - 3.5, color=GRAY)


# =====================================================================
# 1 封面
# =====================================================================
s = prs.slides.add_slide(BLANK)
_rect(s, 0, 0, W, H, RGBColor(0x11, 0x1A, 0x20))
_rect(s, 0, H - Inches(0.22), W, Inches(0.22), ACC)
_tb(s, Inches(0.9), Inches(1.5), Inches(11), Inches(0.5),
    "复杂系统分析 · 期末项目", size=15, color=RGBColor(0x8F, 0xA3, 0xAE))
_tb(s, Inches(0.9), Inches(2.05), Inches(11.4), Inches(2.2),
    "一个底层包的漏洞，\n怎样级联到整个生态？", size=46, bold=True, color=WHITE, line=1.22)
_tb(s, Inches(0.9), Inches(4.35), Inches(11.4), Inches(1.0),
    "PyPI 依赖网络中的影响传播、瓶颈识别与韧性情景实验",
    size=20, color=RGBColor(0xC8, 0xD6, 0xDE))
_tb(s, Inches(0.9), Inches(5.4), Inches(11.4), Inches(1.2),
    "选题 Q4 · 时间切片 2026-11 · 15,780 个包 / 47,944 条依赖边 / 26,810 条安全公告",
    size=13, color=RGBColor(0x7E, 0x93, 0x9F))
_tb(s, Inches(0.9), Inches(6.55), Inches(11), Inches(0.5),
    "答辩人：（按组内分工填写）　　组员：（按组内分工填写）", size=12,
    color=RGBColor(0x6C, 0x80, 0x8C))

# =====================================================================
# 2 一句话结论
# =====================================================================
s = new_slide("三分钟版结论：三个数字", "如果只记三件事，记这三个",
              "对应：results/tables/t22_decomposition.csv, t23_half_thresholds.csv, t24_null_compare.csv")
kv_block(s, Inches(0.6), Inches(1.75), Inches(12.1), [
    ("108.7×", "结构口径 ÷ 版本感知口径（模型内部倍数，非真实风险倍数）"),
    ("41.8 pp", "改用食物网规则会高估的韧性"),
    ("46× → 1.85×", "零模型分解：暴涨绝大部分是「人造巨型环」，真实拓扑只值 1.85×"),
])
bullets(s, Inches(0.7), Inches(3.45), Inches(11.8), [
    ("① 影响面是重尾的；但“结构上会波及”与“真的受害”是两个口径的数",
     "前 0.5% 的 79 个包承担 51% 的潜在波及量；S/V=108.7× 是口径差，"
     "不能说成“真实风险被高估 108.7 倍”，更不能归因于“依赖者升级”。"),
    ("② 软件依赖是「与门」，食物网是「或门」——不能照搬",
     "移除 0.5% 的关键包（78 个），软件规则下 60.5% 的种子包立刻不可构建，食物网规则下只有 18.8%。"),
    ("③ 零模型给了我们一个「被自己推翻」的结论",
     "朴素保度交换看着像 46 倍暴涨，但那是它自己造出的 484 节点巨型环；"
     "用保环零模型剥掉后，真实拓扑只值 1.85 倍。上一版的 38–57 倍已撤回，改写为 1.85 倍。"),
])

# =====================================================================
# 3 选题：从现实问题到科学问题
# =====================================================================
s = new_slide("选题：把一个「大家都知道很严重」的现象变成可测量的量",
              "对应课程「地铁站」案例的同一结构",
              "report/00_选题与研究设计.md §1")
rows = [
    ("现实现象", "Log4Shell、event-stream、xz 后门……少数底层包一出事，整个生态跟着遭殃"),
    ("研究问题", "一个底层包的漏洞，沿依赖链向上传播多远、波及多少包、覆盖多少真实安装量？"),
    ("科学假设", "影响面重尾且集中在少数瓶颈包；拓扑中心性 ≠ 级联影响面；FAIL 规则决定韧性结论"),
    ("分析方法", "有向依赖网络 + 级联可达性 + 多指标瓶颈识别 + 移除/修复情景 + 保度序列零模型"),
]
for i, (k, v) in enumerate(rows):
    y = Inches(1.62 + i * 1.16)
    _rect(s, Inches(0.62), y, Inches(2.0), Inches(0.92), LIGHT)
    _tb(s, Inches(0.72), y + Inches(0.22), Inches(1.9), Inches(0.5), k, size=15, bold=True, color=ACC)
    _tb(s, Inches(2.82), y + Inches(0.16), Inches(9.9), Inches(0.72), v, size=15.5)
_tb(s, Inches(0.62), Inches(6.35), Inches(12), Inches(0.5),
    "关键：题目里的“多远”“多少”本身就是可测量的量（深度 / 广度），而“哪个包最危险”是可排序、可证伪的。",
    size=14, bold=True, color=ACC2)

# =====================================================================
# 4 数据与方法：网络定义
# =====================================================================
s = new_slide("网络怎么建：节点、边、以及必须说清的简化假设",
              "课程要求「解释节点、边与简化假设」",
              "report/02_数据与方法.md §3")
_tb(s, Inches(0.62), Inches(1.6), Inches(5.9), Inches(0.4), "节点 N = S ∪ T", size=17, bold=True)
_tb(s, Inches(0.62), Inches(2.0), Inches(5.9), Inches(1.5),
    "S：下载量榜前 15,000 名（“人们在装什么”）\n    实际取到 14,992（7 个包已从 PyPI 删除）\n"
    "T：闭包里出现但不在榜上的包 → 仅 788 个\n    ⇒ 榜内已闭合 99.5% 的依赖，截断偏差很小",
    size=13.5, color=GRAY)
_tb(s, Inches(0.62), Inches(3.55), Inches(5.9), Inches(0.4), "边 u → v  ⟺  u 的直接依赖包含 v", size=17, bold=True)
_tb(s, Inches(0.62), Inches(3.95), Inches(5.9), Inches(1.9),
    "· 方向 = 依赖流方向；级联传播方向相反\n"
    "· 只保留 DIRECT 边（把 INDIRECT 当边会把图压成团）\n"
    "· 数据源：deps.dev v3 的**已解析闭包**（含每个依赖的确切版本）\n"
    "  —— 这是能做“版本感知判定”的关键，PyPI 的 requires_dist 做不到",
    size=13.5, color=GRAY)
_rect(s, Inches(6.9), Inches(1.6), Inches(5.8), Inches(4.25), LIGHT)
_tb(s, Inches(7.1), Inches(1.75), Inches(5.4), Inches(0.4), "5 条简化假设及其偏差方向",
    size=15, bold=True, color=ACC)
_tb(s, Inches(7.1), Inches(2.2), Inches(5.4), Inches(3.5),
    "边不加权 → 高估弱依赖的影响\n"
    "剔除 extra 可选依赖 → 低估“装了才触发”的漏洞\n"
    "一包一节点（合并多版本）→ 高估结构口径共现\n"
    "不做到方法级可达性 → 我们的“现实影响”仍是上界\n"
    "单层静态快照 → 结论只对该切片成立（用时间外推检验）",
    size=12.5, color=GRAY, line=1.5)
_tb(s, Inches(0.62), Inches(6.1), Inches(12.1), Inches(0.7),
    "时间切片的定义（把“固定切片”变成可审计的事实）：切片 ≡ (榜单月份, 每个包被解析到的具体版本号) —— 逐包落盘，可随时重建。",
    size=13, bold=True, color=ACC2)

# =====================================================================
# 5 结构
# =====================================================================
s = new_slide("结构：入度极度重尾、依赖极浅、而且是异配的",
              "15,780 节点 / 47,944 边 / 密度 1.9e-4",
              "results/tables/t20_structure.csv, t20_degree_stats.csv, t20_hubs.csv；图 f01")
fit_pic(s, "f01_degree_ccdf", Inches(0.45), Inches(1.65), Inches(8.2), Inches(4.0))
kv_block(s, Inches(8.85), Inches(1.7), Inches(3.9), [
    ("0.909", "入度 Gini"), ("2.26", "依赖深度均值"),
])
bullets(s, Inches(8.9), Inches(3.25), Inches(3.9), [
    ("56.4%", "的包没有任何包依赖它"),
    ("1,604", "numpy 的直接依赖者数（最多）"),
    ("−0.148", "度同配性 ⇒ 异配：枢纽被低度包依赖"),
])
_tb(s, Inches(0.6), Inches(5.85), Inches(12.2), Inches(0.8),
    "两个直接后果：① 依赖链很浅（p50=2）⇒ 介数中心性在这张网上几乎没有“过路点”，后面会看到它失效；"
    "② 异配结构 ⇒ 应用→库→基础包是分层的，这正是保护生态的东西（§零模型）。",
    size=12.5, color=GRAY)

# =====================================================================
# 6 广度
# =====================================================================
s = new_slide("影响面：前 0.5% 的包承担了一半以上的潜在波及量",
              "H1 成立（事先声明的证伪线：前 1% 承担 < 50%）",
              "results/tables/t21_top30_by_n_transitive_dependents.csv；图 f02")
fit_pic(s, "f02_cascade_breadth", Inches(0.45), Inches(1.6), Inches(7.6), Inches(3.9))
_rect(s, Inches(8.3), Inches(1.68), Inches(4.45), Inches(3.85), LIGHT)
_tb(s, Inches(8.5), Inches(1.82), Inches(4.1), Inches(0.4), "承担全部级联暴露的比例",
    size=13.5, bold=True, color=ACC)
_tb(s, Inches(8.5), Inches(2.28), Inches(4.1), Inches(3.1),
    "前 0.1%（15 个）　→ 22.4%\n"
    "前 0.5%（78 个）　→ 51.0%\n"
    "前 1.0%（157 个）→ 64.6%\n"
    "前 2.0%（314 个）→ 78.0%\n"
    "前 5.0%（785 个）→ 89.0%\n"
    "————————————\n"
    "加权下载量口径更极端：\n前 1% → 83.8%",
    size=13, color=INK, line=1.6)
_tb(s, Inches(0.6), Inches(5.7), Inches(12.2), Inches(0.9),
    "平均一个包被 9.75 个包传递依赖；最坏 typing-extensions 被 4,316 个种子包依赖 = 整个生态的 31.0%。",
    size=13.5, color=GRAY)

# =====================================================================
# 7 深度 vs 广度
# =====================================================================
s = new_slide("深度和广度是两个独立维度——两种不同的失效模式",
              "“传播多远”不能只看“波及多少”",
              "results/tables/t21_top30_by_cascade_depth.csv；图 f03")
fit_pic(s, "f03_cascade_depth", Inches(0.45), Inches(1.6), Inches(7.4), Inches(3.9))
data = [
    ("又深又广", "typing-extensions：4,316 个依赖者 / 20 跳", ACC),
    ("极广但浅", "idna 3,929 / 11 跳；certifi 3,580 / 11 跳", GRAY),
    ("极深但窄", "cattrs 129 / 19 跳；jsii 91 / 18 跳", ACC2),
    ("AWS CDK 深链", "aws-cdk-* 系列构成 17–20 层的链条", GRAY),
]
for i, (h, b, c) in enumerate(data):
    y = Inches(1.78 + i * 1.05)
    _rect(s, Inches(8.15), y, Inches(4.6), Inches(0.86), LIGHT)
    _tb(s, Inches(8.3), y + Inches(0.08), Inches(4.3), Inches(0.35), h, size=13.5, bold=True, color=c)
    _tb(s, Inches(8.3), y + Inches(0.44), Inches(4.3), Inches(0.35), b, size=11, color=GRAY)
_tb(s, Inches(0.6), Inches(5.75), Inches(12.2), Inches(0.8),
    "治理手段不同：广而浅的枢纽要**隔离/冗余**；深而窄的链条要**打断**（比如 CDK 栈的 20 层嵌套）。",
    size=13.5, color=GRAY)

# =====================================================================
# 8 指标一致性
# =====================================================================
s = new_slide("“哪个包重要”：贵指标不如便宜指标",
              "H2 成立（证伪线：介数与真实影响面的 ρ 与传递依赖者数差 < 0.05）",
              "results/tables/t21_metric_spearman.csv, t25_predictive_power*.csv；图 f04、f09")
fit_pic(s, "f04_metric_spearman", Inches(0.45), Inches(1.6), Inches(5.2), Inches(4.1))
fit_pic(s, "f09_predictive_power", Inches(5.9), Inches(1.6), Inches(7.0), Inches(4.1))
_tb(s, Inches(0.6), Inches(5.85), Inches(12.2), Inches(0.8),
    "入度 / 传递依赖者数 / 级联深度 / PageRank 四者互相 ρ ≥ 0.93（冗余）；"
    "介数与它们只有 ρ ≈ 0.40，预测真实影响面时 ρ = 0.10（严格子集下 −0.11，符号都反了）。",
    size=13, color=GRAY)

# =====================================================================
# 9 介数为什么失效
# =====================================================================
s = new_slide("介数中心性在这张网上基本失效——这是一个可解释的失败",
              "依赖深度 p50 = 2 ⇒ 几乎没有“过路点”",
              "results/tables/t21_top30_by_betweenness.csv")
_tb(s, Inches(0.62), Inches(1.6), Inches(12.0), Inches(0.5),
    "介数榜 Top-3 的荒谬之处：", size=16, bold=True)
tbl_rows = [
    ("apache-airflow", "91", "91", "介数第一，但影响面只有 91（不到 typing-extensions 的 1/50）"),
    ("requests", "1,550", "3,120", "真枢纽，但介数只排第二且绝对值极小（4.5e-5）"),
    ("apache-airflow-core", "2", "93", "入度只有 2，却进了介数前三 ⇒ 说明它只是“长链过路点”"),
]
for i, (nm, ind, br, note) in enumerate(tbl_rows):
    y = Inches(2.15 + i * 1.05)
    _rect(s, Inches(0.62), y, Inches(12.1), Inches(0.86), LIGHT)
    _tb(s, Inches(0.78), y + Inches(0.12), Inches(3.0), Inches(0.4), nm, size=15, bold=True)
    _tb(s, Inches(3.9), y + Inches(0.12), Inches(1.6), Inches(0.4), f"入度 {ind}", size=13, color=GRAY)
    _tb(s, Inches(5.5), y + Inches(0.12), Inches(1.8), Inches(0.4), f"影响面 {br}", size=13, color=GRAY)
    _tb(s, Inches(7.4), y + Inches(0.12), Inches(5.2), Inches(0.4), note, size=12, color=INK)
_tb(s, Inches(0.62), Inches(5.45), Inches(12.1), Inches(1.4),
    "原因：本网络 56.4% 的节点入度为 0、依赖深度中位数只有 2 跳，绝大多数“依赖路径”长度为 1–2，"
    "根本不经过第三个节点。真正有非零介数的，只是少数长链上的节点（AWS CDK 那一串）。\n"
    "⇒ 介数识别的是“长链上的过路点”，不是“影响面大”。课程示例里度中心性打败介数/PageRank，这里是同一个现象的另一种表现。",
    size=13, color=GRAY, line=1.5)

# =====================================================================
# 10 三级口径
# =====================================================================
s = new_slide("三个口径之间差多少，以及这些差**不能**说明什么",
              "每个百分比都写出分母；口径收缩 ≠ 因果归因",
              "results/tables/t22_decomposition.csv, t22_mechanism.csv；图 f05、f06")
fit_pic(s, "f05_struct_vs_version", Inches(0.45), Inches(1.6), Inches(8.4), Inches(3.75))
_rect(s, Inches(9.1), Inches(1.68), Inches(3.65), Inches(3.6), LIGHT)
_tb(s, Inches(9.28), Inches(1.82), Inches(3.3), Inches(0.4), "三级口径", size=14, bold=True, color=ACC)
_tb(s, Inches(9.28), Inches(2.25), Inches(3.3), Inches(2.9),
    "S 结构上界\n　1,188,370 对\n\nF 反事实未修复\n　1,075,784（90.49%）\n\n"
    "V 现实（版本感知）\n　10,931（0.92%）\n\nS/V = 108.7×（模型内部倍数）\n"
    "加权下载量口径 = 367.6×",
    size=12.5, color=INK, line=1.35)
_tb(s, Inches(0.6), Inches(5.55), Inches(12.2), Inches(1.25),
    "两个收缩率、两个分母、两件不同的事：\n"
    "① S→F = (S−F)/S = 9.47%（分母 S）—— 只抹掉 OSV 的 fixed 边界，收缩很小 ⇒ "
    "绝大多数结构命中本来就落在 ≥introduced 一侧。这一步与“依赖者是否升级”无关。\n"
    "② F→V = (F−V)/F = 98.98%（分母 F）—— 命中收缩到“落在脆弱区间内”。单区间 advisory 上"
    "这个差集恒等于“resolved ≥ fixed”，是一条定义恒等式，不是机制发现。\n"
    "⚠ 不能据此说“真实风险被高估 108.7 倍”，也不能说“依赖者完成了 98.98% 的升级”——我们没有时间序列。\n"
    "唯一带机制含义的自检：25 条至今无补丁的 advisory，结构/现实比只有 1.23×。",
    size=11, color=GRAY, line=1.3)

# =====================================================================
# 11 榜单对照
# =====================================================================
s = new_slide("两个榜单几乎不重叠：这就是结论本身",
              "“结构上会波及 3,929 个包”不能直接拿去做优先级",
              "results/tables/t22_worst_advisories.csv, t22_worst_advisories_real.csv")
top = [("结构口径最吓人", [
    ("PYSEC-2024-60 · idna", "结构 3,929　现实 1"),
    ("GHSA-43fp-rhv2-5gv8 · certifi", "结构 3,580　现实 0"),
    ("PYSEC-2023-135 · certifi", "结构 3,580　现实 0"),
    ("PYSEC-2026-215 · idna", "结构 3,929　现实 2"),
], ACC),
    ("现实口径最严重", [
    ("PYSEC-2022-43059 · aiohttp", "结构 751　现实 716"),
    ("PYSEC-2025-189 · torch", "结构 284　现实 268"),
    ("PYSEC-2025-193 · torch", "结构 284　现实 268"),
    ("PYSEC-2022-43059 之外 · 均为未修区间", "见 t22_worst_advisories_real"),
], ACC2)]
for i, (title, items, color) in enumerate(top):
    x = Inches(0.62 + i * 6.25)
    _rect(s, x, Inches(1.62), Inches(5.9), Inches(4.1), LIGHT)
    _tb(s, x + Inches(0.18), Inches(1.75), Inches(5.5), Inches(0.4), title, size=15, bold=True, color=color)
    for j, (a, b) in enumerate(items):
        yy = Inches(2.25 + j * 0.85)
        _tb(s, x + Inches(0.18), yy, Inches(5.5), Inches(0.38), a, size=12.5, bold=True)
        _tb(s, x + Inches(0.18), yy + Inches(0.34), Inches(5.5), Inches(0.35), b, size=11.5, color=GRAY)
_tb(s, Inches(0.62), Inches(5.95), Inches(12.2), Inches(0.8),
    "86.2% 的 advisory 在今天的解析结果里一个人都不影响；过近似中位数 31.4×，p90 250×。",
    size=13.5, bold=True, color=ACC)

# =====================================================================
# 12 韧性：AND vs OR
# =====================================================================
s = new_slide("同样的移除，两种失效规则给出完全相反的韧性结论",
              "H4 成立（证伪线：q ≤ 0.1 内两规则差 < 5 个百分点；实测 q=0.5% 时差 41.8 个百分点）",
              "results/tables/t23_half_thresholds.csv, t23_curves_summary.csv；图 f07")
fit_pic(s, "f07_resilience_rules", Inches(0.45), Inches(1.55), Inches(8.6), Inches(3.9))
_rect(s, Inches(9.25), Inches(1.62), Inches(3.5), Inches(3.85), LIGHT)
_tb(s, Inches(9.42), Inches(1.75), Inches(3.2), Inches(0.4), "q = 0.5%（79 个包）", size=13, bold=True, color=ACC)
_tb(s, Inches(9.42), Inches(2.2), Inches(3.2), Inches(3.0),
    "AND（软件真实语义）\n　按传递依赖者 → S = 0.378\n　随机 → S = 0.961\n\n"
    "OR（食物网语义）\n　按传递依赖者 → S = 0.796\n　随机 → S = 0.995\n\n"
    "靶向 vs 随机差 18.8 倍\n规则差 41.8 个百分点",
    size=12, color=INK, line=1.4)
_tb(s, Inches(0.6), Inches(5.7), Inches(12.2), Inches(0.9),
    "软件依赖是「与门」（缺一个依赖就废），食物网是「或门」（还有别的猎物）。"
    "把次级灭绝模型直接搬过来，会让软件生态看起来远比实际稳健。",
    size=13.5, color=GRAY)

# =====================================================================
# 13 修复 vs 移除
# =====================================================================
s = new_slide("“修复”与“移除”是两种性质不同的干预：只报一个指标会得出反向结论",
              "题目可做方向 #3：模拟“关键包被修复”与“关键包被移除”两种情景",
              "results/tables/t23_patch_vs_remove.csv；图 f10")
fit_pic(s, "f10_patch_vs_remove", Inches(0.45), Inches(1.6), Inches(8.4), Inches(3.8))
_rect(s, Inches(9.1), Inches(1.68), Inches(3.65), Inches(3.65), LIGHT)
_tb(s, Inches(9.28), Inches(1.82), Inches(3.3), Inches(0.4), "处理 10 个瓶颈包", size=13.5, bold=True, color=ACC)
_tb(s, Inches(9.28), Inches(2.3), Inches(3.3), Inches(2.9),
    "情景 P「修复」\n　暴露量 ↓ 29.3%\n　可用性代价 0\n\n情景 R「移除」\n　暴露量 ↓ 54.2%\n"
    "　但 50.1% 的种子包\n　彻底不可构建\n\n⇒ 移除的“暴露下降”\n　是因为受害者已经死了",
    size=12, color=INK, line=1.4)
_tb(s, Inches(0.6), Inches(5.62), Inches(12.2), Inches(1.0),
    "正确读法：修复 10 个瓶颈包可以在零可用性代价下把生态总暴露量削掉 29%；"
    "移除同样 10 个包，暴露量虽掉 54%，但一半的包不能用了。",
    size=13.5, color=GRAY)

# =====================================================================
# 14 零模型
# =====================================================================
s = new_slide("★ 我们把上一版的一条主结论撤回了：零模型的 46× 里有 96% 是伪影",
              "评审问：随机图自己长出来的大环算不算数？—— 实测：不算数的地方比想的多",
              "results/tables/t24_null_compare.csv, t24_null_curves.csv, t24_null_swaps.json；图 f08")
fit_pic(s, "f08_null_model", Inches(0.45), Inches(1.55), Inches(7.3), Inches(3.35))
_rect(s, Inches(7.95), Inches(1.6), Inches(4.75), Inches(3.25), LIGHT)
_tb(s, Inches(8.1), Inches(1.72), Inches(4.45), Inches(3.0),
    "最大强连通分量（20 次实现中位数）\n"
    "　真实图　　　　　  8 个节点\n"
    "　朴素保度交换　**484** 个节点\n"
    "　保环保度交换　　  2 个节点\n\n"
    "反向可达均值\n"
    "　真实　　　　　11.08\n"
    "　保环族　　　　20.53 ± 1.53　→ 1.85×\n"
    "　朴素族　　　 526.92 ± 94.85　→ 47.6×",
    size=11.5, color=INK, line=1.35)
_tb(s, Inches(0.62), Inches(5.0), Inches(12.1), Inches(1.75),
    "分解（这是本页唯一的结论）：真实 → 保环族 = **1.85×**（真实拓扑的贡献）；"
    "保环族 → 朴素族 = **25.7×**（纯人造巨型环的贡献）。\n"
    "⇒ 「分层结构把级联压低一个半到两个数量级」**撤回**，改写为「1.85 倍量级的有限效应」。\n"
    "⇒ 「随机故障下比同度序列随机图鲁棒近 6 倍」**撤回**：q=1% 时 0.920 vs 0.871±0.105（差 4.9 个百分点）。\n"
    "仍然成立的两条：枢纽排名几乎不变（保环族 ρ=0.980）；靶向移除下三者几乎重合（0.375 / 0.348 / 0.351）。",
    size=11.5, color=GRAY, line=1.35)

# =====================================================================
# 15 双重高风险
# =====================================================================
s = new_slide("“结构上关键”和“历史上多灾”不是同一件事",
              "富集 7.6 倍，但整体 Spearman 只有 0.12–0.13 ⇒ 需要两套治理手段",
              "results/tables/t25_dual_risk_crosstab.csv, t25_dual_risk_top50.csv；图 f11")
fit_pic(s, "f11_dual_risk", Inches(0.45), Inches(1.6), Inches(6.4), Inches(4.15))
dd = read_csv_safe(TBL / "t25_dual_risk_top50.csv").head(11)
rows = [("包", "依赖者", "公告数")]
for r in dd.itertuples(index=False):
    rows.append((r.name, f"{int(r.n_transitive_dependents)}", f"{int(r.n_advisories)}"))
tb = s.shapes.add_table(len(rows), 3, Inches(7.15), Inches(1.62), Inches(5.6),
                        Inches(0.31 * len(rows))).table
for i, row in enumerate(rows):
    for j, v in enumerate(row):
        c = tb.cell(i, j)
        c.text = str(v)
        for p in c.text_frame.paragraphs:
            p.alignment = PP_ALIGN.RIGHT if j else PP_ALIGN.LEFT
            for run in p.runs:
                run.font.size = Pt(10.5)
                run.font.name = "Microsoft YaHei"
                run.font.bold = (i == 0)
                run.font.color.rgb = ACC if i == 0 else INK
_tb(s, Inches(7.15), Inches(5.35), Inches(5.6), Inches(1.3),
    "39 个双重高风险包（结构瓶颈 ∧ 多灾）；\n"
    "反例：typing-extensions 4,316 个依赖者、0 条公告（关键不多灾）；\n"
    "　　　django 339 条公告、402 个依赖者（多灾不关键）。",
    size=11.5, color=GRAY, line=1.45)

# =====================================================================
# 16 可靠性
# =====================================================================
s = new_slide("可靠性：重复、误差、敏感性、稳健性",
              "对应课程「如何判断结果是否可靠」",
              "report/02_数据与方法.md §6、report/03_结果.md §7、§8")
bullets(s, Inches(0.7), Inches(1.6), Inches(5.9), [
    ("随机种子", "随机移除重复 30 次，报均值 ± s.d.；所有 seed 写在 config.yaml"),
    ("误差分析", "相关系数全部给 bootstrap 95% CI；介数报采样 k=2000"),
    ("敏感性", "PageRank α、介数采样 k、移除比例网格、排序协议均可改"),
], size=15.5, gap=1.28)
bullets(s, Inches(6.9), Inches(1.6), Inches(5.9), [
    ("稳健性（换口径）", "闭包口径 vs 图口径，Spearman 0.971"),
    ("稳健性（换规则）", "AND vs OR 两种失效规则同时报告"),
    ("稳健性（换网络）", "两族保度序列零模型各 20 次实现；度序列由 assert 逐点校验"),
], size=15.5, gap=1.28)
_rect(s, Inches(0.7), Inches(5.6), Inches(11.9), Inches(1.0), LIGHT)
_tb(s, Inches(0.9), Inches(5.72), Inches(11.6), Inches(0.8),
    "数据质量全部记账、不静默丢弃：闭包回退 525 个（3.5%）、版本无法解析 8.25%、"
    "版本判定不可判 5.0%、图口径截断约 2%。",
    size=12.5, color=ACC)

# =====================================================================
# 17 局限
# =====================================================================
s = new_slide("局限：三条会改变结论方向的，必须写在结论旁边",
              "“结论是否超出证据”",
              "report/04_讨论与结论.md §3")
lims = [
    ("没有做方法级可达性分析",
     "版本脆弱 ≠ 代码真的调用了脆弱函数。所以我们的 10,931 对仍然是上界，108.7× 是“版本级过近似”。", ACC),
    ("现实口径 = “今天全新解析”，不是“所有已部署环境”",
     "大量用户 pin 在旧版本，那部分存量风险不在 V 口径里。不能拿它论证“生态已经安全”。", ACC),
    ("“级联深度”用最长路径",
     "在没有真实攻击时序时，最长路径是潜在上限，不是“实际会同时发生”的路径。", ACC),
    ("会改变数字但不改变方向",
     "边不加权（高估弱依赖）、剔除 extra（低估）、一包一节点（高估共现）、单一时间切片。", GRAY),
]
for i, (h, b, c) in enumerate(lims):
    y = Inches(1.62 + i * 1.22)
    _rect(s, Inches(0.62), y, Inches(12.1), Inches(1.02), LIGHT)
    _rect(s, Inches(0.62), y, Inches(0.08), Inches(1.02), c)
    _tb(s, Inches(0.88), y + Inches(0.10), Inches(11.6), Inches(0.36), h, size=14, bold=True, color=c)
    _tb(s, Inches(0.88), y + Inches(0.50), Inches(11.6), Inches(0.48), b, size=11.5, color=GRAY)

# =====================================================================
# 18 结论与建议
# =====================================================================
s = new_slide("三条可操作的建议（以及对题目四个问题的回答）",
              "对应 report/04_讨论与结论.md §5",
              "report/04_讨论与结论.md")
recs = [
    ("① 别只报“结构受影响依赖数”",
     "在我们的数据上它是现实的 108.7 倍，且 86.2% 的公告在今天一个人都不影响。至少加一步版本判定——材料我们已经证明可得。"),
    ("② 区分“关键”与“多灾”",
     "typing-extensions（4,316 依赖者、0 公告）需要冗余与镜像；pillow（215 公告、816 依赖者）需要补丁跟进自动化。混成一个风险分会两边都做不好。"),
    ("③ 优先“打断链条”和“隔离枢纽”，而不是祈祷没人盯上枢纽",
     "对照实验说明：靶向移除下真实网络与随机网络一样脆弱；分层结构的保护只对随机故障有效。所以单点冗余才是能改变结论的手段。"),
]
for i, (h, b) in enumerate(recs):
    y = Inches(1.65 + i * 1.42)
    _rect(s, Inches(0.62), y, Inches(12.1), Inches(1.2), LIGHT)
    _tb(s, Inches(0.85), y + Inches(0.12), Inches(11.6), Inches(0.4), h, size=15, bold=True, color=ACC)
    _tb(s, Inches(0.85), y + Inches(0.58), Inches(11.6), Inches(0.55), b, size=12, color=GRAY)
_tb(s, Inches(0.62), Inches(6.05), Inches(12.1), Inches(0.9),
    "对题目四个问题：(a) 平均波及 11 个包、最坏 4,316 个 / 深度最多 20 跳；(b) Top-10 瓶颈包已给出；"
    "(c) 结构口径÷版本感知口径 = 108.7×（模型内部倍数；拆成 9.47% + 98.98% 两段，"
    "分母分别是 S 与 F，均为口径描述而非因果归因）；"
    "(d) 修复 10 个包零代价削 29% 暴露，移除同 10 个则半数包报废；另测「保护」这一第三动作。",
    size=11.5, color=INK, line=1.45)

# =====================================================================
# 19 分工 + 复现
# =====================================================================
s = new_slide("分工与复现", "四人分工按实际产出对齐；一键脚本 + 唯一参数真源",
              "run_all.ps1, config/config.yaml, report/05_答辩讲稿.md 文末分工表")
_tb(s, Inches(0.62), Inches(1.62), Inches(5.9), Inches(0.4), "正式分工（与讲稿一致）", size=15, bold=True, color=ACC)
_tb(s, Inches(0.62), Inches(2.05), Inches(5.9), Inches(3.4),
    "A｜数据与网络　　　src/01、02、10、11、versions.py\n"
    "　　　　　　　　　（抓取、清洗、建图、数据质量记账）\n"
    "B｜中心性与榜单　　src/21、22、25、27\n"
    "　　　　　　　　　（广度/深度/中心性、三级口径、榜单口径）\n"
    "C｜级联规则与情景　src/23、24、26、28\n"
    "　　　　　　　　　（AND/OR、修复/移除/保护、零模型与排序协议）\n"
    "D｜可视化与验证　　src/12、29、30、40、tests/、50\n"
    "　　　　　　　　　（图表、案例路径、边抽检、PPT、数字核对）",
    size=11.5, color=INK, line=1.5)
_tb(s, Inches(6.9), Inches(1.62), Inches(5.9), Inches(0.4), "复现", size=15, bold=True, color=ACC2)
_tb(s, Inches(6.9), Inches(2.05), Inches(5.9), Inches(3.4),
    "powershell -File .\\run_all.ps1　　　全流程（含 API 抓取）\n"
    "powershell -File .\\run_all.ps1 -SkipFetch　只重跑分析\n\n"
    "第一步是小图手算测试（55 项断言），跑不过就不许往下算\n\n"
    "config/config.yaml = 唯一参数真源\n"
    "  · 时间切片定义\n"
    "  · 网络口径（seed_top_n / 边方向 / 叶子纳入）\n"
    "  · 实验参数（移除比例、重复次数、seed、marker 环境）\n\n"
    "src/50_verify_numbers.py 机器核对正文数字与结果表",
    size=11.5, color=GRAY, line=1.45)
_rect(s, Inches(0.62), Inches(5.55), Inches(12.1), Inches(1.1), LIGHT)
_tb(s, Inches(0.85), Inches(5.68), Inches(11.6), Inches(0.85),
    "数据来源（全部公开、无需密钥）：PyPI 月下载量榜（hugovk 镜像 pypistats）· "
    "deps.dev v3 API（Google osv-scanner 团队）· OSV.dev PyPI 漏洞库打包。\n"
    "依赖：Python 3.13 + networkx / pandas / numpy / scipy / matplotlib / packaging / pyarrow / pyyaml。",
    size=11.5, color=GRAY, line=1.45)

# =====================================================================
# 20 Q&A
# =====================================================================
s = prs.slides.add_slide(BLANK)
_rect(s, 0, 0, W, H, RGBColor(0x11, 0x1A, 0x20))
_rect(s, 0, H - Inches(0.22), W, Inches(0.22), ACC)
_tb(s, Inches(0.9), Inches(2.55), Inches(11), Inches(1.2), "Q & A", size=54, bold=True, color=WHITE)
_tb(s, Inches(0.9), Inches(3.85), Inches(11.5), Inches(2.2),
    "一个底层包的漏洞，怎样级联到整个生态？\n"
    "平均 11 个包、最坏 4,316 个、最深 20 跳；\n"
    "而“结构上会波及”与“真的受害”是两个不同口径的数——\n"
    "这两件事的区别，恰好是整个供应链安全里最容易被忽略的一步。",
    size=17, color=RGBColor(0xC8, 0xD6, 0xDE), line=1.6)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
log(f"生成完成：{OUT}（{len(prs.slides.__iter__.__self__._sldIdLst)} 页）")
log.close()
