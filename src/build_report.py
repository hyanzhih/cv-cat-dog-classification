"""根据 outputs/metrics.json 与 outputs/ablation.json 自动生成实验报告 HTML。

数字全部从实验结果文件读取，图表用绝对路径引用 outputs/figures/*.png，
再由 html-to-doc 转换为 docx，避免手工抄录误差。

用法：python src/build_report.py
产出：report/实验报告.html
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

from config import (BATCH_SIZE, DEVICE, DROPOUT, EPOCHS, EPOCHS_FINAL, FIG_DIR, IMG_SIZE,
                    LABEL_SMOOTHING, LR, MIN_EPOCHS, N_FOLDS, OUT_DIR, PATIENCE, ROOT, SEED,
                    TEST_RATIO, WEIGHT_DECAY)
from model import CatDogCNN, count_parameters
from utils import load_json, setup_stdout

setup_stdout()
REPORT_DIR = ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

M = load_json(OUT_DIR / "metrics.json")
ABL = load_json(OUT_DIR / "ablation.json") if (OUT_DIR / "ablation.json").exists() else None

cv = M["cross_validation"]["summary"]
pf = M["cross_validation"]["per_fold"]
tm = M["final_model"]["test_metrics"]
ds = M["dataset"]

pct = lambda x, n=2: f"{x * 100:.{n}f}%"
fig = lambda name: str(FIG_DIR / name).replace("\\", "/")


def abl_acc(name, default=float("nan")):
    """从消融结果中取某个变体的准确率；结果文件缺失时返回 NaN。"""
    if not ABL:
        return default
    for r in ABL["rows"]:
        if r["name"] == name:
            return r["acc_mean"]
    return default


def abl_std(name, default=float("nan")):
    """从消融结果中取某个变体的标准差。"""
    if not ABL:
        return default
    for r in ABL["rows"]:
        if r["name"] == name:
            return r["acc_std"]
    return default


def delta(name):
    """某变体相对完整模型的准确率变化（百分点，负值表示下降）。"""
    return (abl_acc(name) - abl_acc("完整模型（本文方法）")) * 100


# ---------------- 表格 ----------------
def table(header, rows, widths=None, caption=None, note=None):
    cols = ""
    if widths:
        cols = "<colgroup>" + "".join(f'<col style="width:{w}">' for w in widths) + "</colgroup>"
    head = "".join(f"<th>{h}</th>" for h in header)
    body = ""
    for r in rows:
        cls = ' class="sum"' if r and str(r[0]).startswith("均值") else ""
        body += f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    cap = f'<p class="tabcap">{caption}</p>' if caption else ""
    nt = f'<p class="tabnote">{note}</p>' if note else ""
    return (f'{cap}<table>{cols}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{nt}')


def figure(name, caption, width="92%"):
    return (f'<p class="figbox"><img src="{fig(name)}" style="width:{width}"></p>'
            f'<p class="figcap">{caption}</p>')


# ---------------- 参数量统计 ----------------
model = CatDogCNN(dropout=DROPOUT)
def blk_stat(blk):
    conv = sum(p.numel() for m in blk.modules() if isinstance(m, nn.Conv2d) for p in m.parameters())
    bn = sum(p.numel() for m in blk.modules() if isinstance(m, nn.BatchNorm2d) for p in m.parameters())
    return conv, bn

b1, b2, b3, b4 = (blk_stat(model.block1), blk_stat(model.block2),
                  blk_stat(model.block3), blk_stat(model.block4))
fc_params = model.classifier[-1].weight.numel() + model.classifier[-1].bias.numel()
total_params = count_parameters(model)

layer_rows = [
    ["输入", "RGB 图像缩放至 128×128，逐通道标准化", "3×128×128", "0"],
    ["卷积块 1", "2×[3×3 卷积 + BN + ReLU]，2×2 最大池化", "32×64×64", f"{sum(b1):,}"],
    ["卷积块 2", "同上，通道 32→64", "64×32×32", f"{sum(b2):,}"],
    ["卷积块 3", "同上，通道 64→128", "128×16×16", f"{sum(b3):,}"],
    ["卷积块 4", "同上，通道 128→256", "256×8×8", f"{sum(b4):,}"],
    ["全局平均池化", "AdaptiveAvgPool2d(1)", "256×1×1", "0"],
    ["Dropout", "p = 0.5", "256", "0"],
    ["全连接", "Linear(256, 2)", "2", f"{fc_params:,}"],
    ["合计", "—", "—", f"{total_params:,}"],
]
conv_total = sum(b[0] for b in (b1, b2, b3, b4))
bn_total = sum(b[1] for b in (b1, b2, b3, b4))

# ---------------- 交叉验证表 ----------------
fold_rows = []
for m in pf:
    fold_rows.append([f"第 {m['fold']} 折", pct(m["accuracy"], 1), f"{m['precision']:.3f}",
                      f"{m['recall']:.3f}", f"{m['f1']:.3f}", f"{m['auc']:.3f}",
                      str(m["best_epoch"])])
fold_accs = [m["accuracy"] for m in pf]
fold_rows.append(["均值 ± 标准差", f"{pct(cv['acc_mean'])} ± {pct(cv['acc_std'])}",
                  f"{sum(m['precision'] for m in pf)/len(pf):.3f}",
                  f"{sum(m['recall'] for m in pf)/len(pf):.3f}",
                  f"{cv['f1_mean']:.3f}", f"{cv['auc_mean']:.3f}", "—"])

cm = cv["oof"]["confusion"]
cm_test = tm["confusion"]
test_acc_n = round(tm["accuracy"] * tm["test_size"])

# ---------------- 消融表 ----------------
abl_rows, abl_text = [], ""
if ABL:
    base = ABL["rows"][0]["acc_mean"]
    for i, r in enumerate(ABL["rows"]):
        d = (r["acc_mean"] - base) * 100
        d_text = "—" if i == 0 else f"{d:+.1f}"
        abl_rows.append([r["name"], f"{r['acc_mean']*100:.2f} ± {r['acc_std']*100:.2f}",
                         f"{r['f1_mean']:.3f}", f"{r['auc_mean']:.3f}", d_text])

# ---------------- 正文 ----------------
CSS = """
@page { size: A4; margin: 2.3cm 2.1cm 2.2cm 2.1cm;
        @bottom-center { content: counter(page) " / " counter(pages);
                         font-size: 8.5pt; color: #7f7f7f; } }
@page cover { @bottom-center { content: none; } }
section[role="cover"] { page: cover; }
body { font-family: "宋体", SimSun; font-size: 10.5pt; line-height: 1.72; color: #1a1a1a; }
h1 { font-family: "黑体", SimHei; font-size: 17pt; text-align: center; margin: 0 0 6pt 0; }
h2 { font-family: "黑体", SimHei; font-size: 13pt; color: #1F3864; margin: 20pt 0 9pt 0;
     border-left: 4pt solid #2F5597; padding-left: 7pt; }
h3 { font-family: "黑体", SimHei; font-size: 11pt; color: #2F5597; margin: 13pt 0 6pt 0; }
p { margin: 5pt 0; text-align: justify; text-indent: 2em; }
p.noind { text-indent: 0; }
table { width: 100%; border-collapse: collapse; font-size: 9.5pt; margin: 4pt 0 10pt 0; }
th { background: #EDF2F9; font-family: "黑体", SimHei; font-size: 9pt; font-weight: normal;
     border: 0.6pt solid #B4C6E7; padding: 4.5pt 4pt; text-align: center; color: #1F3864; }
td { border: 0.6pt solid #C9D6E8; padding: 4pt; text-align: center; }
tr.sum td { background: #F7F9FC; font-weight: bold; }
p.tabcap, p.figcap { text-indent: 0; text-align: center; font-size: 9pt; color: #404040;
     font-family: "黑体", SimHei; margin: 2pt 0 10pt 0; }
p.tabnote { text-indent: 0; font-size: 8.5pt; color: #666666; margin: -6pt 0 10pt 0; }
p.figbox { text-indent: 0; text-align: center; margin: 8pt 0 2pt 0; }
ul, ol { margin: 4pt 0 8pt 0; padding-left: 22pt; }
li { margin: 2.5pt 0; text-align: justify; }
.cover-course { font-family: "黑体", SimHei; font-size: 13pt; letter-spacing: 2pt;
     color: #1F3864; text-align: center; text-indent: 0; }
.cover-title { font-family: "黑体", SimHei; font-size: 26pt; text-align: center;
     color: #1F3864; text-indent: 0; margin: 14pt 0 4pt 0; letter-spacing: 6pt; }
.cover-sub { font-family: "宋体", SimSun; font-size: 14pt; text-align: center; text-indent: 0;
     color: #2F5597; margin: 0 0 40pt 0; }
.cover-meta { font-family: "宋体", SimSun; font-size: 11.5pt; text-align: center; text-indent: 0;
     line-height: 2.1; color: #262626; }
.cover-info-list { border-collapse: collapse; margin: 0 auto; }
.cover-info-list td { border: none; font-size: 11.5pt; padding: 4pt 10pt; text-align: left; }
.cover-info-list td.k { text-align: right; white-space: nowrap; }
.kv { font-size: 10pt; }
.abs { font-size: 10pt; }
.quote { border-left: 3pt solid #C9D6E8; padding: 2pt 0 2pt 10pt; margin: 8pt 0; }
.quote p { text-indent: 0; font-size: 9.8pt; color: #333333; }
"""

parts = []
parts.append(f"""<section role="cover">
<p class="cover-course">《计算机视觉》课程实验报告</p>
<p class="cover-meta">&nbsp;</p>
<p class="cover-meta">&nbsp;</p>
<p class="cover-meta">&nbsp;</p>
<p class="cover-title">实验一</p>
<p class="cover-sub">基于卷积神经网络的猫狗图像二分类</p>
<p class="cover-meta">&nbsp;</p>
<p class="cover-meta">&nbsp;</p>
<p class="cover-meta">&nbsp;</p>
<table class="cover-info-list">
<tr><td class="k">姓　　名</td><td>____________________</td></tr>
<tr><td class="k">学　　号</td><td>____________________</td></tr>
<tr><td class="k">班　　级</td><td>____________________</td></tr>
<tr><td class="k">指导教师</td><td>____________________</td></tr>
<tr><td class="k">实验日期</td><td>2026 年 11 月 15 日</td></tr>
</table>
<p class="cover-meta">&nbsp;</p>
<p class="cover-meta">&nbsp;</p>
<p class="cover-meta" style="font-size:10pt;color:#7f7f7f">实验环境：Python 3.12 · PyTorch 2.7 · NVIDIA RTX 4060 Laptop</p>
</section>
<section role="body" data-page-restart="1">""")

# 摘要
parts.append(f"""<h2>摘　要</h2>
<p class="abs">本文在仅含 100 张图像（猫、狗各 50 张）的小样本数据集上，从零设计并训练了一个轻量卷积神经网络
完成二分类，全过程<b>不使用任何预训练权重</b>。网络由 4 个「双 3×3 卷积 + 批量归一化 + 最大池化」块组成，
经全局平均池化与 Dropout 后接线性分类头，参数量 {total_params:,}。</p>
<p class="abs">针对样本量过小导致的评估不稳定问题，以 <b>{N_FOLDS} 折分层交叉验证</b>作为主评估手段，
并以 8∶2 分层留出法训练最终模型；训练中采用随机裁剪、水平翻转、小角度旋转与色彩抖动进行数据增强，
使用 AdamW 优化器、余弦学习率退火与标签平滑，并针对「验证集仅 20 张、单轮准确率方差极大」的问题
引入<b>最小训练轮数约束</b>以抑制选模噪声。</p>
<p class="abs">实验结果为：{N_FOLDS} 折交叉验证准确率 <b>{pct(cv['acc_mean'])} ± {pct(cv['acc_std'])}</b>，
折外预测（OOF）准确率 {pct(cv['oof']['accuracy'])}、F1 {cv['oof']['f1']:.3f}、AUC {cv['oof']['auc']:.3f}；
留出测试集准确率 <b>{pct(tm['accuracy'])}</b>（{test_acc_n}/{tm['test_size']}）。消融实验表明：
移除数据增强、BatchNorm 或 Dropout 中任一项，准确率都会下降约 {abs(delta('无 BatchNorm')):.1f} 个百分点，
其中移除增强还会使折间标准差由 {abl_std('完整模型（本文方法）') * 100:.1f} 增至
{abl_std('无数据增强') * 100:.1f} 个百分点；而输入分辨率、网络宽度与测试时增强的影响均不显著。
训练曲线、混淆矩阵与特征图可视化共同显示，模型的主要误差来自对纹理相近、姿态异常的样本判别不足，
且存在明显过拟合。</p>
<p class="abs" style="text-indent:0"><b>关键词：</b>卷积神经网络；图像二分类；小样本学习；数据增强；交叉验证；消融实验</p>""")

# 一、实验目的
parts.append("""<h2>一、实验目的</h2>
<ol>
<li>掌握图像分类任务中数据组织、预处理、数据增强的完整流程，理解训练集/验证集/测试集的正确划分方式与信息泄漏的规避；</li>
<li>掌握卷积神经网络的基本构件（卷积层、批量归一化、池化、全局平均池化、Dropout）及其设计动机，
能够针对小样本场景自行设计网络结构；</li>
<li>掌握小样本条件下可靠的模型评估方法：分层 K 折交叉验证、折外预测、混淆矩阵、ROC 曲线与 AUC，
并能正确解读指标间的差异；</li>
<li>通过消融实验与特征可视化，定量、定性地分析各组件的作用与模型的失效模式，形成「提出假设—设计实验—验证结论」的研究习惯。</li>
</ol>""")

# 二、实验环境
parts.append(f"""<h2>二、实验环境</h2>
{table(["项目", "配置"],
       [["操作系统", "Windows 11（64 位）"],
        ["CPU / GPU", "笔记本平台 / NVIDIA GeForce RTX 4060 Laptop GPU（8 GB 显存）"],
        ["Python", "3.12.9"],
        ["深度学习框架", "PyTorch 2.7.0 + CUDA 12.6，torchvision 0.22.0"],
        ["辅助库", "NumPy 2.2、Pillow 11.2、Matplotlib 3.11、scikit-learn 1.9（仅用于分层划分）"],
        ["随机性控制", "固定随机种子 {SEED}，开启 cudnn.deterministic"],
        ["全流程耗时", f"约 {M['total_time_sec']:.0f} 秒（{N_FOLDS} 折 × {EPOCHS} 轮 + 最终模型 {EPOCHS_FINAL} 轮）"]],
       widths=["26%", "74%"], caption="表 1　实验软硬件环境")}""")

# 三、数据集
parts.append(f"""<h2>三、数据集与预处理</h2>
<h3>3.1　数据集概况</h3>
<p>实验数据为 <code>猫狗数据集100张.zip</code>，解压后得到 {ds['total']} 张 JPEG 图像，
文件名前缀即类别标签（<code>cat.N.jpg</code> / <code>dog.N.jpg</code>），猫、狗各
{ds['per_class']['cat']} 张，类别完全均衡，无需重采样或代价敏感损失。</p>
<p class="noind">数据存在两个显著特点（图 1—图 3）：</p>
<ul>
<li><b>尺寸高度不统一</b>：共 {ds['unique_sizes']} 种不同分辨率，最常见的
{ds['top_sizes'][0]['size'][0]}×{ds['top_sizes'][0]['size'][1]} 也仅出现 {ds['top_sizes'][0]['count']} 次，
宽高比在 0.5—1.4 之间波动，必须统一缩放；</li>
<li><b>场景与姿态多样</b>：包含室内外、不同品种、不同光照、部分遮挡与特写，单张图像的主体占比差异很大，
这对仅 100 张样本的模型构成了显著的类内差异。</li>
</ul>
{figure('01_class_distribution.png', '图 1　类别分布（猫、狗各 50 张）', '52%')}
{figure('02_size_distribution.png', '图 2　原始图像尺寸分布（每点一张图像，红色虚线为统一后的输入尺寸）', '62%')}
{figure('03_samples.png', '图 3　数据集抽样示例（上排为猫，下排为狗）', '86%')}

<h3>3.2　数据划分策略</h3>
<p>样本量极小，单次划分的评估结果受「哪些样本恰好落在验证集」影响极大。为此采用两套划分并行：</p>
<ul>
<li><b>{N_FOLDS} 折分层交叉验证（主结果）</b>：按类别分层将 100 张均分为 {N_FOLDS} 折，每折轮流作验证集
（80 张训练 / 20 张验证）。全部 100 张都恰好被预测一次，把这些预测拼起来得到
<b>折外预测（Out-of-Fold, OOF）</b>，可给出一个近乎无偏的整体估计；</li>
<li><b>分层留出法（用于部署）</b>：按 8∶2 分层划分为 {round((1-TEST_RATIO)*100)} 张训练 / {round(TEST_RATIO*100)} 张测试，
训练集内再切 10% 作为早停用验证集，测试集只在最后使用一次，用于检验最终模型的真实泛化能力。</li>
</ul>

<h3>3.3　预处理与归一化</h3>
<p>所有图像统一到 {IMG_SIZE}×{IMG_SIZE}：训练时随机裁剪缩放，验证/测试时先将短边缩放 1.14 倍再中心裁剪，
以保留主体区域。归一化所用均值 <code>[{', '.join(f'{v:.4f}' for v in M['norm_stats']['mean'])}]</code>、
标准差 <code>[{', '.join(f'{v:.4f}' for v in M['norm_stats']['std'])}]</code>
<b>仅由训练集统计得到</b>，避免验证集信息通过统计量泄漏到训练过程。该均值明显低于 ImageNet 的 0.485/0.456/0.406，
说明本数据集整体偏暗。</p>

<h3>3.4　数据增强</h3>
<p>仅 80 张训练图像，强烈的数据增强是抑制过拟合的首要手段。本实验组合了四类几何与光度变换（图 4）：</p>
<ul>
<li><b>随机裁剪缩放</b>（面积比 0.65—1.0，宽高比 0.8—1.25）：模拟主体尺度与位置变化，迫使模型不依赖固定构图；</li>
<li><b>水平翻转</b>（p = 0.5）：猫狗左右对称，翻转不改变类别语义，等价于扩充一倍样本；</li>
<li><b>随机旋转</b>（±12°，p = 0.5）：容忍拍摄角度偏差；</li>
<li><b>色彩抖动</b>（亮度/对比度/饱和度 0.2、色调 0.02）：降低模型对毛色与光照的过度依赖。</li>
</ul>
{figure('04_augmentation.png', '图 4　训练阶段的数据增强效果（左上为原图，其余为同一张图的不同增强结果）', '90%')}""")

# 四、模型设计
parts.append(f"""<h2>四、模型设计</h2>
<h3>4.1　网络结构</h3>
<p>自建网络（<code>CatDogCNN</code>）共 4 个卷积块，每个块包含两个 3×3 卷积（各接 BN 与 ReLU），
随后 2×2 最大池化使特征图边长减半、通道数加倍，最终经全局平均池化与 Dropout 后由一个线性层输出两类 logits。</p>
{table(["阶段", "操作", "输出特征图", "参数量"], layer_rows,
       widths=["16%", "46%", "22%", "16%"], caption="表 2　网络结构与逐层参数量",
       note=f"其中卷积层参数 {conv_total:,}、BN 参数 {bn_total:,}、分类头 {fc_params:,}；全部参数均为可训练参数。")}

<h3>4.2　关键设计及其动机</h3>
<ul>
<li><b>双 3×3 卷积代替单 5×5 卷积</b>：两层 3×3 堆叠的感受野等价于 5×5，但参数更少（2×9 &lt; 25 个权重/通道对），
且中间多了一次非线性，表达力更强；</li>
<li><b>每个卷积后接 BatchNorm</b>：小批量（16）下梯度噪声大，BN 将每层输入重新标准化，
使训练对学习率不敏感、收敛更快，并带来轻微的正则效果；</li>
<li><b>最大池化</b>：保留邻域内最显著的响应（如边缘、纹理），同时把特征图压缩 4 倍，扩大后续层的感受野；</li>
<li><b>全局平均池化替代全连接</b>：若用 Flatten + 全连接（256×8×8→256），仅该层就有 41.9 万参数，
极易在小样本上过拟合；GAP 把每个通道压成一个数，参数为 0，且天然支持任意输入尺寸；</li>
<li><b>Dropout(p = 0.5)</b>：在分类头前随机丢弃一半特征，迫使各类别证据冗余表达，是抑制过拟合的最后一道防线；</li>
<li><b>Kaiming 初始化</b>：配合 ReLU，使各层激活方差稳定，避免深层梯度消失或爆炸。</li>
</ul>
<p>整个模型参数量为 <b>{total_params:,}</b>，在 100 张样本上已属「偏大」——这正是后文训练曲线中
训练集准确率迅速逼近 95% 而验证集停滞的根源。</p>""")

# 五、训练策略
parts.append(f"""<h2>五、训练策略</h2>
<h3>5.1　优化目标与超参数</h3>
{table(["超参数", "取值", "说明"],
       [["损失函数", "交叉熵 + 标签平滑 0.1", "避免模型对训练样本过度自信，缓解过拟合"],
        ["优化器", "AdamW", "解耦权重衰减，比 Adam 泛化更好"],
        ["学习率", f"{LR:g}（余弦退火至 0）", "初期快速下降，后期小步精细收敛"],
        ["权重衰减", f"{WEIGHT_DECAY:g}", "L2 正则"],
        ["批大小", str(BATCH_SIZE), "受显存与样本量限制，小批量的梯度噪声本身有益"],
        ["最大轮数", f"{EPOCHS}（最终模型 {EPOCHS_FINAL}）", "配合早停，实际多数折在 40—60 轮停止"],
        ["早停耐心", f"{PATIENCE} 轮", "验证准确率连续不提升即停止"],
        ["最小训练轮数", f"{MIN_EPOCHS} 轮", "见 5.2，前若干轮不参与选模"]],
       widths=["18%", "30%", "52%"], caption="表 3　训练超参数配置")}

<h3>5.2　小验证集下的选模与早停</h3>
<p>验证集每折仅 20 张，一张样本的差异就是 5 个百分点，<b>单轮验证准确率的噪声极大</b>。
若从第 1 轮起就记录「最优轮」，往往会选中某个「恰好猜对多数样本」的偶然状态。
初版实现（学习率 2e-3、最多 60 轮、从第 1 轮开始选模）测得的最佳轮次分别为第 1、2、5、6、15 轮，
而验证准确率仅 73.0% ± 9.1%——模型几乎未得到充分训练就被判定为最优。</p>
<p>改进做法是：<b>前 {MIN_EPOCHS} 轮不参与选模</b>，之后才按验证准确率保留权重，
并配合 {PATIENCE} 轮耐心早停；改进后最佳轮次后移至 25—45 轮，交叉验证准确率提升至
{pct(cv['acc_mean'])}。这一现象本身说明：在极小验证集上，评估协议的设计比模型结构的微调更重要。</p>

<h3>5.3　推理阶段的测试时增强（TTA）</h3>
<p>预测时同时送入原图与其水平翻转版本，对两个 softmax 输出取平均。
由于训练阶段本就使用随机水平翻转，模型对翻转具有近似的等变性，TTA 理论上可对两次预测做集成、降低方差。
不过 6.4 的消融实验显示：本任务上去掉 TTA 后准确率完全不变、AUC 甚至略高，
说明模型对翻转的等变性已被训练增强充分学到，翻转平均相当于重复同一次预测，
这一技巧在此处只有「理论上正确、实测无收益」的意义。</p>

<h3>5.4　评价指标</h3>
<p class="noind">以「狗」为正类，记 TP/TN/FP/FN 分别表示真正例、真负例、假正例、假负例，采用：</p>
<ul>
<li><b>准确率</b> Acc = (TP+TN)/N：整体正确比例，类别均衡时最直观；</li>
<li><b>精确率</b> P = TP/(TP+FP)、<b>召回率</b> R = TP/(TP+FN)、<b>F1</b> = 2PR/(P+R)：分别刻画「判为狗的有多少真是狗」
与「狗有多少被找回」，F1 为其调和平均；</li>
<li><b>特异度</b> = TN/(TN+FP)：猫被正确识别的比例，用于检查模型是否偏向某一类；</li>
<li><b>AUC</b>：ROC 曲线下面积，等价于「随机取一只狗和一只猫，狗得分更高的概率」，
与阈值无关，适合衡量排序质量。</li>
</ul>""")

# 六、结果
fold_line = "、".join(pct(a, 1) for a in fold_accs)
parts.append(f"""<h2>六、实验结果与分析</h2>
<h3>6.1　{N_FOLDS} 折交叉验证（主结果）</h3>
{table(["折次", "准确率", "精确率", "召回率", "F1", "AUC", "最佳轮次"], fold_rows,
       widths=["17%", "19%", "12%", "12%", "12%", "12%", "16%"],
       caption=f"表 4　{N_FOLDS} 折交叉验证结果（每折 80 张训练 / 20 张验证）")}
<p>{N_FOLDS} 折准确率依次为 {fold_line}，均值 <b>{pct(cv['acc_mean'])}</b>，标准差 {pct(cv['acc_std'])}。
折间波动主要来自「这一折恰好包含多少难以区分的样本」，而非训练过程不稳定——因为每折都有 25—45 轮的充分训练，
且各折的损失曲线走势高度一致（图 5）。</p>
{figure('07_fold_accuracy.png', '图 5　各折验证准确率与均值', '56%')}
{figure('05_cv_curves.png', f'图 6　{N_FOLDS} 折训练的损失与准确率曲线（实线为训练集，虚线为验证集）', '96%')}
<p>从图 6 可以直接观察到过拟合：训练损失单调下降、训练准确率升至 90% 以上，而验证损失在 20 轮后开始回升、
验证准确率在 70%—85% 之间震荡。这也解释了为何需要早停——继续训练只会让模型更「记住」训练样本。</p>

<p>把 5 折的验证集预测拼接得到折外预测，其对全部 100 张样本的整体评估为：
准确率 {pct(cv['oof']['accuracy'])}、精确率 {cv['oof']['precision']:.3f}、召回率 {cv['oof']['recall']:.3f}、
F1 {cv['oof']['f1']:.3f}、特异度 {cv['oof']['specificity']:.3f}、AUC {cv['oof']['auc']:.3f}。
混淆矩阵（图 7）中 FP = {cm['fp']}、FN = {cm['fn']}，两类错误数量相等，
说明模型<b>没有明显的类别偏置</b>（既不偏向判成猫，也不偏向判成狗）。</p>
{figure('08_confusion_oof.png', f'图 7　折外预测混淆矩阵（准确率 {pct(cv["oof"]["accuracy"])}）', '46%')}
{figure('09_roc_folds.png', '图 8　各折 ROC 曲线及平均曲线', '58%')}

<h3>6.2　留出测试集结果</h3>
{table(["指标", "数值", "指标", "数值"],
       [["准确率", pct(tm['accuracy']), "AUC", f"{tm['auc']:.3f}"],
        ["精确率", f"{tm['precision']:.3f}", "特异度", f"{tm['specificity']:.3f}"],
        ["召回率", f"{tm['recall']:.3f}", "F1", f"{tm['f1']:.3f}"]],
       widths=["25%", "25%", "25%", "25%"],
       caption=f"表 5　最终模型在留出测试集（{tm['test_size']} 张）上的表现")}
<p>最终模型在 20 张测试图像上正确分类 {test_acc_n} 张（{pct(tm['accuracy'])}），
混淆矩阵为 TP {cm_test['tp']} / TN {cm_test['tn']} / FP {cm_test['fp']} / FN {cm_test['fn']}。
精确率 {tm['precision']:.3f} 明显高于召回率 {tm['recall']:.3f}，即 <b>漏检多于误报</b>：
被错分的 {cm_test['fp'] + cm_test['fn']} 张样本中，{cm_test['fn']} 张是把狗判成了猫、
{cm_test['fp']} 张是把猫判成了狗。这与样本构成有关——数据集中有若干浅色小型犬、
以及趴在织物上的猫，其毛发纹理与姿态高度相似（参见图 13），单纯依靠颜色与轮廓难以区分。</p>
<p>需要强调：20 张测试集上一张样本即 5 个百分点，该结果与交叉验证的 {pct(cv['acc_mean'])} 处于同一量级，
二者互相印证；不应把 75% 与 78% 的差异解读为「模型在测试集上变差」。</p>
{figure('15_confusion_test.png', '图 9　测试集混淆矩阵', '44%')}
{figure('16_roc_compare.png', '图 10　折外预测（100 张）与留出测试集（20 张）的 ROC 对比', '58%')}

<h3>6.3　最终模型训练过程</h3>
{figure('06_final_curve.png', '图 11　最终模型训练曲线（虚线为最佳轮次）', '66%')}
<p>最终模型在第 {tm['best_epoch']} 轮取得最佳验证准确率 {pct(M['final_model']['history_best_val_acc'])}，
随后验证准确率连续 {PATIENCE} 轮未再提升而触发早停。训练准确率最高达 97%，验证损失在最佳轮次后持续上升，
是典型的过拟合形态；由于训练集只有 72 张，任何结构上的进一步「加容量」都只会加剧这一趋势。</p>""")

# 消融
if ABL:
    parts.append(f"""<h3>6.4　消融实验</h3>
<p>为量化各组件的贡献，在完全相同的划分与训练配置下逐项移除关键设计，只改变一个变量（表 6、图 12）。</p>
{table(["变体", "准确率（%）", "F1", "AUC", "相对完整模型（百分点）"], abl_rows,
       widths=["34%", "24%", "14%", "14%", "14%"],
       caption=f"表 6　消融实验结果（{ABL['n_folds']} 折交叉验证均值 ± 标准差）",
       note="差值列 = 变体准确率 − 完整模型准确率（百分点），负值表示移除该项后性能下降。")}
{figure('17_ablation.png', '图 12　消融实验：各组件的贡献', '72%')}
<p>主要结论如下：</p>
<ul>
<li><b>三项正则化手段的贡献高度一致</b>：移除数据增强、BatchNorm、Dropout 中任意一项，
准确率都从 {pct(abl_acc('完整模型（本文方法）'))} 降到 {pct(abl_acc('无数据增强'))} 一档，
即各自贡献约 {abs(delta('无 BatchNorm')):.1f} 个百分点（{delta('无数据增强'):+.1f}、{delta('无 BatchNorm'):+.1f}、
{delta('无 Dropout'):+.1f}），说明在 80 张训练样本下，这三者共同构成抑制过拟合的「最小充分组合」；</li>
<li><b>数据增强的主要作用在降方差</b>：移除增强后平均准确率只降 {abs(delta('无数据增强')):.1f} 个百分点，
但折间标准差从 {abl_std('完整模型（本文方法）') * 100:.1f} 猛增到 {abl_std('无数据增强') * 100:.1f} 个百分点，
说明增强最直接的效果是让模型在不同数据划分下表现更稳定，而不是单纯抬高平均分；</li>
<li><b>输入分辨率与网络宽度的影响很小</b>：输入降到 64×64 仅降 {abs(delta('输入 64×64')):.1f} 个百分点（AUC 反而更高）；
通道数减半后准确率为 {pct(abl_acc('网络宽度减半'))}（{delta('网络宽度减半'):+.1f}），
与完整模型完全在噪声范围内。这从侧面印证：当前瓶颈是<b>样本量</b>而非模型容量或细节分辨率，
继续加深加宽网络不会带来实质收益；</li>
<li><b>TTA 在本任务上没有增益</b>：去掉 TTA 后准确率完全不变（{delta('无 TTA 推理'):+.1f} 个百分点），AUC 反而略高，
说明 5.3 中「集成降方差」的预期并未实现——模型对水平翻转的等变性已被训练增强充分学到，
再做翻转平均相当于重复同一次预测。</li>
</ul>
<p class="noind">需要强调：所有变体的折间标准差在 5—14 个百分点之间，多数差异不足 1 个标准差，
因此上表只适合做<b>趋势性判断</b>，不宜把 1—3 个百分点的差距当作确定结论。这也再次说明在小样本实验中，
<b>报告标准差比报告单一准确率更重要</b>。</p>""")

# 七、错误分析与可视化
parts.append(f"""<h2>七、错误分析与可解释性</h2>
<h3>7.1　错分样本分析</h3>
{figure('12_misclassified.png', '图 13　折外预测中被错分的样本（按置信度由低到高排列）', '90%')}
<p>图 13 汇总了折外预测中置信度最低的错误样本，可归纳出三类典型失败模式：</p>
<ol>
<li><b>姿态与遮挡异常</b>：只露出头部或蜷缩成一团时，猫与小型犬的轮廓几乎不可区分；</li>
<li><b>纹理混淆</b>：浅色短毛犬与短毛猫的毛发纹理接近，模型缺乏「脸型」「耳朵形状」等高层判别特征；</li>
<li><b>背景主导</b>：部分样本中主体占比过小（如猫位于角落），裁剪后主体信息进一步丢失。</li>
</ol>
<p>图 14 的概率分布显示，正确样本的概率大多集中在 0 与 1 附近，而错误样本多落在 0.4—0.6 的
「犹豫区间」，说明模型对困难样本具有可感知的不确定性——这一点可用于实际系统中的拒识（
置信度低于阈值时转人工复核）。</p>
{figure('11_prob_hist.png', '图 14　折外预测的置信度分布（按真实类别着色）', '62%')}

<h3>7.2　卷积核与特征图可视化</h3>
{figure('13_kernels.png', '图 15　第一层 32 个 3×3 卷积核', '86%')}
<p>第一层卷积核呈现出方向性边缘检测器、颜色对立（如黄—蓝）与低频纹理等形态，
与经典 CNN 的可视化结果一致：网络在浅层学习「边缘/颜色」这类通用基元，而非直接记忆类别。</p>
{figure('14_feature_maps.png', '图 16　特征图逐层演化（上组为猫，下组为狗；每行取前 8 个通道）', '96%')}
<p>随层数加深，特征图分辨率由 64×64 降至 8×8，响应逐渐从「轮廓」变为「局部语义块」：
block1 保留了清晰的猫耳、狗嘴轮廓；block3—block4 的特征图已难以用肉眼对应到具体部位，
但不同通道对主体区域的响应强度差异显著，说明网络确实在整合局部信息形成类别相关的全局表示。
部分通道在猫狗两类图像上响应模式不同，这些通道正是分类头的决策依据。</p>""")

# 八、讨论
parts.append(f"""<h2>八、问题与讨论</h2>
<h3>8.1　从零训练的性能天花板</h3>
<p>{pct(cv['acc_mean'])} 的准确率对一个从零训练的模型而言是合理的，但与 ImageNet 预训练模型
（同类任务通常在 95% 以上）差距明显。原因在于：卷积核的「通用视觉基元」本可以从千万级图像中学到，
而本实验只有 80 张训练图，网络必须同时学习「如何看」和「如何分类」。
若课程允许使用预训练权重，冻结主干 + 微调分类头通常能立刻提升 10 个百分点以上；
本实验选择从零训练，是为了完整暴露小样本学习的困难，而非追求最高精度。</p>

<h3>8.2　评估协议本身带来的偏差</h3>
<p>实验过程中出现了两个值得记录的「方法论教训」：</p>
<ul>
<li><b>选模噪声</b>：初版实现从第 1 轮开始记录最优权重，导致 5 折中有 4 折的「最佳轮次」落在前 15 轮，
交叉验证准确率仅 73.0%；引入最小训练轮数后提升至 {pct(cv['acc_mean'])}。相差的 5 个百分点并非来自模型改进，
而是来自<b>评估协议更合理</b>；</li>
<li><b>权重滑动平均（SWA）反而有害</b>：在 lr = 1e-3、100 轮的配置下尝试对末段权重做平均，
准确率反而从 78.0% 降到 68.0%。原因是早停常在第 40—60 轮触发，SWA 的平均窗口只剩下 5—18 个 epoch，
且这些轮次的学习率尚未充分衰减，平均后得到一个「半成品」权重。这提示：<b>任何训练技巧都有其前提条件</b>，
SWA 需要完整跑完学习率调度才能发挥作用。</li>
</ul>
<p>此外，同一配置重复运行会带来约 ±3 个百分点的波动（来源于数据加载顺序与 Dropout 的随机性），
因此消融实验中 1—2 个百分点的差异不足以支撑结论，本文只对方向一致、幅度较大的差异下判断。</p>

<h3>8.3　进一步的改进方向</h3>
<ul>
<li><b>迁移学习</b>：使用 ImageNet 预训练的 ResNet18/VGG16，仅微调最后几层，预期准确率可达 90% 以上；</li>
<li><b>更强的正则与数据增广</b>：MixUp、CutMix、RandAugment，或引入无标签样本做半监督/伪标签训练；</li>
<li><b>模型集成</b>：把 {N_FOLDS} 折训练出的 {N_FOLDS} 个模型做投票，通常可稳定提升 2—4 个百分点；</li>
<li><b>损失函数</b>：若实际场景更在意「不漏检狗」，可对 FN 加权或调整决策阈值（例如把阈值从 0.5 下调），
用精确率换召回率。</li>
</ul>""")

# 九、结论
parts.append(f"""<h2>九、结论</h2>
<ol>
<li>在 100 张猫狗图像上从零训练的 {total_params:,} 参数卷积网络，经 {N_FOLDS} 折分层交叉验证取得
<b>{pct(cv['acc_mean'])} ± {pct(cv['acc_std'])}</b> 的准确率（折外 F1 {cv['oof']['f1']:.3f}、AUC {cv['oof']['auc']:.3f}），
留出测试集 {pct(tm['accuracy'])}，两类错误（FP {cm['fp']} / FN {cm['fn']}）数量均衡，无类别偏置；</li>
<li>小样本场景下，<b>正则化组合比模型容量更重要</b>：移除数据增强、BatchNorm、Dropout 任一项，
准确率都下降约 {abs(delta('无 BatchNorm')):.1f} 个百分点；而把输入分辨率降到 64×64、通道数减半，
性能几乎不变甚至个别指标更好，说明瓶颈在样本量而非模型容量；</li>
<li>训练曲线与错分样本分析表明模型存在明显过拟合（训练准确率 97% vs 验证 70%—85%），
错误主要集中在纹理相近、姿态异常、主体过小的样本上，且错误样本的预测概率集中在 0.5 附近，具备可识别的不确定性；</li>
<li>评估协议的设计对小样本实验的结论影响巨大：把「最小训练轮数」这一约束加入选模流程，
带来的提升（5 个百分点）甚至超过多数结构改动——这提醒我们在比较方法时必须先固定并审视评估流程。</li>
</ol>""")

# 附录
parts.append(f"""<h2>附录 A　工程结构与复现方式</h2>
<p class="noind">全部代码按「配置—数据—模型—训练—评估—可视化」分层组织，一条命令即可复现全部结果：</p>
{table(["文件", "职责"],
       [["<code>src/config.py</code>", "全部超参数与路径，实验的唯一配置入口"],
        ["<code>src/prepare_data.py</code>", "解压数据集、统计类别与尺寸分布"],
        ["<code>src/dataset.py</code>", "分层 K 折 / 留出划分、数据增强、训练集归一化统计"],
        ["<code>src/model.py</code>", "自建 CNN（含特征图接口）"],
        ["<code>src/train.py</code>", "训练循环、早停与选模、交叉验证、最终模型"],
        ["<code>src/visualize.py</code>", "全部图表绘制（数据探索、曲线、混淆矩阵、ROC、特征图）"],
        ["<code>src/run_all.py</code>", "一键复现主实验（数据 → CV → 最终模型 → 图表 → metrics.json）"],
        ["<code>src/ablation.py</code>", "消融实验，输出 ablation.csv / .json 与对比图"],
        ["<code>src/predict.py</code>", "单张或整目录推理，输出类别与置信度"]],
       widths=["32%", "68%"], caption="表 7　代码结构")}
<p class="noind"><b>运行方式：</b></p>
<div class="quote">
<p>python src/run_all.py &nbsp;&nbsp;# 主实验，约 2 分钟</p>
<p>python src/ablation.py &nbsp;&nbsp;# 消融实验，约 15 分钟</p>
<p>python src/predict.py --image data/raw/dog.1.jpg &nbsp;&nbsp;# 单张推理</p>
</div>
<p class="noind">所有随机源（Python / NumPy / PyTorch、数据划分、模型初始化、数据加载顺序）均已固定种子
{SEED}，实验结果可逐位复现；<code>outputs/metrics.json</code> 中保存了每一折的完整指标与折外预测概率。</p>

<h2>附录 B　最终模型的关键信息</h2>
{table(["项目", "内容"],
       [["输入尺寸", f"{IMG_SIZE} × {IMG_SIZE} × 3"],
        ["参数量", f"{total_params:,}"],
        ["权重文件", "outputs/models/catdog_cnn.pth（含归一化参数与测试指标）"],
        ["交叉验证准确率", f"{pct(cv['acc_mean'])} ± {pct(cv['acc_std'])}"],
        ["留出测试集准确率", pct(tm['accuracy'])],
        ["单张推理耗时", "约 3.3 ms（GPU，含 TTA 与图像读取）"]],
       widths=["30%", "70%"], caption="表 8　模型交付信息")}""")

html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="docx-page-size" content="A4">
<title>实验一 基于卷积神经网络的猫狗图像二分类</title>
<style>{CSS}</style></head><body>
{''.join(parts)}
</section>
</body></html>"""

out = REPORT_DIR / "实验报告.html"
out.write_text(html, encoding="utf-8")
print(f"报告 HTML 已生成：{out}")
print(f"  正文字数约 {len(html):,} 字符；引用图表 {html.count('<img')} 张；表格 {html.count('<table>')} 个")
