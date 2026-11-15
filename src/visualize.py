"""可视化模块：数据探索、训练过程、评估结果与模型可解释性图表。

所有图统一保存在 outputs/figures/，供实验报告直接引用。
"""
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from config import (C_ACCENT, C_GRAY, C_MAIN, CLASSES, CLASS_NAMES_CN, DEVICE, FIG_DIR,
                    IMG_SIZE)
from utils import roc_auc, roc_curve_points, setup_plot_style

setup_plot_style()

CN = [CLASS_NAMES_CN[c] for c in CLASSES]  # ["猫", "狗"]


def _save(fig, name: str, out_dir: Path = FIG_DIR):
    path = Path(out_dir) / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return str(path)


# ---------------- 数据探索 ----------------
def plot_class_distribution(summary: dict, out_dir: Path = FIG_DIR):
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    counts = [summary["per_class"][c] for c in CLASSES]
    bars = ax.bar(CN, counts, color=[C_MAIN, C_ACCENT], width=0.55)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.8, str(c), ha="center", fontsize=11)
    ax.set_ylabel("样本数")
    ax.set_title(f"类别分布（共 {summary['total']} 张）")
    ax.set_ylim(0, max(counts) * 1.25)
    return _save(fig, "01_class_distribution.png", out_dir)


def plot_size_distribution(paths, out_dir: Path = FIG_DIR):
    ws, hs = [], []
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
        ws.append(w)
        hs.append(h)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.scatter(ws, hs, s=26, alpha=0.75, color=C_MAIN, edgecolor="white", linewidth=0.6)
    ax.axhline(IMG_SIZE, color=C_ACCENT, ls="--", lw=1, label=f"统一输入 {IMG_SIZE}×{IMG_SIZE}")
    ax.axvline(IMG_SIZE, color=C_ACCENT, ls="--", lw=1)
    ax.set_xlabel("图像宽度 / px")
    ax.set_ylabel("图像高度 / px")
    ax.set_title(f"原始图像尺寸分布（{len(set(zip(ws, hs)))} 种尺寸）")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, "02_size_distribution.png", out_dir)


def plot_samples(paths, labels, n: int = 8, out_dir: Path = FIG_DIR):
    idx = np.linspace(0, len(paths) - 1, n).astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.9))
    for ax, i in zip(axes.ravel(), idx):
        with Image.open(paths[i]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{CLASS_NAMES_CN[CLASSES[labels[i]]]}  ({Path(paths[i]).name})", fontsize=8)
        ax.axis("off")
    fig.suptitle("数据集中随机抽样样本", fontsize=12)
    return _save(fig, "03_samples.png", out_dir)


def plot_augmentation(path, label, transform, n: int = 6, out_dir: Path = FIG_DIR):
    with Image.open(path) as im:
        img = im.convert("RGB")
    # 原图按短边居中裁成正方形，保证与增强样本拼图尺寸一致
    w, h = img.size
    s = min(w, h)
    img_sq = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.2))
    axes.ravel()[0].imshow(img_sq)
    axes.ravel()[0].set_title(f"原图（{CLASS_NAMES_CN[CLASSES[label]]}，居中裁切显示）", fontsize=8.5)
    axes.ravel()[0].axis("off")
    for ax in list(axes.ravel())[1:]:
        aug = transform(img)
        arr = aug.permute(1, 2, 0).numpy()
        # 反归一化用于显示
        arr = (arr * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        ax.imshow(arr)
        ax.set_title("增强样本", fontsize=8.5)
        ax.axis("off")
    for ax in list(axes.ravel())[1 + n:]:
        ax.axis("off")
    fig.suptitle("训练阶段的数据增强效果（随机裁剪 / 翻转 / 旋转 / 色彩抖动）", fontsize=11)
    fig.tight_layout()
    return _save(fig, "04_augmentation.png", out_dir)


# ---------------- 训练过程 ----------------
def plot_cv_curves(histories, out_dir: Path = FIG_DIR, best_epochs=None):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    for i, h in enumerate(histories, start=1):
        ep = h["epoch"]
        axes[0].plot(ep, h["train_loss"], lw=1.1, alpha=0.75, label=f"第{i}折 训练")
        axes[0].plot(ep, h["val_loss"], lw=1.1, ls="--", alpha=0.75, label=f"第{i}折 验证")
        axes[1].plot(ep, h["train_acc"], lw=1.1, alpha=0.75)
        axes[1].plot(ep, h["val_acc"], lw=1.1, ls="--", alpha=0.75)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("交叉熵损失"); axes[0].set_title("损失曲线（5 折交叉验证）")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("准确率"); axes[1].set_title("准确率曲线（5 折交叉验证）")
    axes[1].set_ylim(0, 1.02)
    axes[0].legend(fontsize=7, ncol=2, framealpha=0.85)
    fig.tight_layout()
    return _save(fig, "05_cv_curves.png", out_dir)


def plot_final_curve(hist, out_dir: Path = FIG_DIR):
    fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
    ep = hist["epoch"]
    ax1.plot(ep, hist["train_loss"], color=C_MAIN, lw=1.6, label="训练损失")
    ax1.plot(ep, hist["val_loss"], color=C_ACCENT, lw=1.6, label="验证损失")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("交叉熵损失")
    ax2 = ax1.twinx()
    ax2.plot(ep, hist["train_acc"], color=C_MAIN, lw=1.2, ls=":", label="训练准确率")
    ax2.plot(ep, hist["val_acc"], color=C_ACCENT, lw=1.2, ls=":", label="验证准确率")
    ax2.set_ylabel("准确率"); ax2.set_ylim(0, 1.05); ax2.grid(False)
    if hist["best_epoch"]:
        ax1.axvline(hist["best_epoch"], color=C_GRAY, ls="--", lw=1)
        ax1.annotate(f"最佳 epoch={hist['best_epoch']}\n验证准确率 {hist['best_val_acc']*100:.1f}%",
                     xy=(hist["best_epoch"], ax1.get_ylim()[1] * 0.75), fontsize=8,
                     color=C_GRAY, ha="right", va="bottom")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    ax1.set_title("最终模型训练过程")
    fig.tight_layout()
    return _save(fig, "06_final_curve.png", out_dir)


def plot_fold_accuracy(per_fold, out_dir: Path = FIG_DIR):
    accs = [m["accuracy"] for m in per_fold]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(1, len(accs) + 1)
    ax.bar(x, accs, color=C_MAIN, width=0.6)
    ax.axhline(np.mean(accs), color=C_ACCENT, ls="--", lw=1.2,
               label=f"均值 {np.mean(accs)*100:.2f}%")
    for xi, a in zip(x, accs):
        ax.text(xi, a + 0.015, f"{a*100:.1f}%", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels([f"第{i}折" for i in x])
    ax.set_ylim(0, 1.12); ax.set_ylabel("验证准确率"); ax.set_title("各折验证准确率")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, "07_fold_accuracy.png", out_dir)


# ---------------- 评估 ----------------
def plot_confusion(matrix, title, name, out_dir: Path = FIG_DIR):
    m = np.array([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(3.9, 3.5))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() * 1.15 if m.max() else 1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center", fontsize=13,
                    color="white" if m[i, j] > m.max() * 0.6 else "#1a1a1a")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"预测{CN[0]}", f"预测{CN[1]}"])
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"真实{CN[0]}", f"真实{CN[1]}"])
    ax.set_title(title, fontsize=11)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, name, out_dir)


def plot_roc(curves, out_dir: Path = FIG_DIR, name="10_roc.png"):
    """curves: [(y_true, y_prob, label), ...]，画每条曲线并对均值曲线做插值。"""
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    colors = ["#2F5597", "#C00000", "#548235", "#BF8F00", "#7030A0", "#008B8B"]
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    for i, (y, p, label) in enumerate(curves):
        fpr, tpr, _ = roc_curve_points(y, p)
        ax.plot(fpr, tpr, lw=1.2, alpha=0.8, color=colors[i % len(colors)],
                label=f"{label} (AUC={roc_auc(y, p):.3f})")
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
    if len(curves) >= 3:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        _trapz = getattr(np, "trapezoid", np.trapz)
        ax.plot(mean_fpr, mean_tpr, color="black", lw=2.0,
                label=f"平均 ROC (AUC={_trapz(mean_tpr, mean_fpr):.3f})")
    ax.plot([0, 1], [0, 1], color=C_GRAY, ls=":", lw=1)
    ax.set_xlabel("假正率 FPR"); ax.set_ylabel("真正率 TPR")
    ax.set_title("ROC 曲线")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)
    return _save(fig, name, out_dir)


def plot_prob_hist(y_true, y_prob, title, name="11_prob_hist.png", out_dir: Path = FIG_DIR):
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    bins = np.linspace(0, 1, 11)
    ax.hist(y_prob[np.array(y_true) == 0], bins=bins, alpha=0.75, color=C_MAIN, label=f"真实{CN[0]}")
    ax.hist(y_prob[np.array(y_true) == 1], bins=bins, alpha=0.75, color=C_ACCENT, label=f"真实{CN[1]}")
    ax.axvline(0.5, color="black", ls="--", lw=1)
    ax.set_xlabel("预测为「狗」的概率"); ax.set_ylabel("样本数")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, name, out_dir)


def plot_misclassified(paths, labels, probs, n: int = 8, out_dir: Path = FIG_DIR,
                       name="12_misclassified.png"):
    wrong = [i for i in range(len(labels)) if int(probs[i] >= 0.5) != int(labels[i])]
    if not wrong:
        return None
    wrong = sorted(wrong, key=lambda i: abs(probs[i] - 0.5))[:n]
    cols = min(4, len(wrong))
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.9 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), wrong):
        with Image.open(paths[i]) as im:
            ax.imshow(im.convert("RGB"))
        true_cn = CLASS_NAMES_CN[CLASSES[labels[i]]]
        pred_cn = CLASS_NAMES_CN[CLASSES[int(probs[i] >= 0.5)]]
        ax.set_title(f"真实:{true_cn}  预测:{pred_cn}\np(狗)={probs[i]:.2f}", fontsize=8.5)
        ax.axis("off")
    fig.suptitle(f"折外预测中被错分的样本（共 {len(wrong)} 张）", fontsize=11)
    fig.tight_layout()
    return _save(fig, name, out_dir)


# ---------------- 消融 ----------------
def plot_ablation(rows, out_dir: Path = FIG_DIR):
    names = [r["name"] for r in rows][::-1]
    accs = np.array([r["acc_mean"] for r in rows])[::-1]
    stds = np.array([r["acc_std"] for r in rows])[::-1]
    colors = [C_MAIN if n.startswith("完整") else "#8FAADC" for n in names]
    fig, ax = plt.subplots(figsize=(6.8, 0.55 * len(rows) + 1.6))
    y = np.arange(len(names))
    ax.barh(y, accs * 100, xerr=stds * 100, color=colors, height=0.55,
            error_kw=dict(ecolor=C_GRAY, lw=1, capsize=3))
    for yi, a, s in zip(y, accs * 100, stds * 100):
        ax.text(a + s + 1.0, yi, f"{a:.1f}", va="center", fontsize=8.5, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("5 折交叉验证准确率 / %")
    ax.set_xlim(0, max(accs * 100 + stds * 100) * 1.12)
    ax.set_title("消融实验：各组件的贡献")
    return _save(fig, "17_ablation.png", out_dir)


# ---------------- 可解释性 ----------------
@torch.no_grad()
def plot_first_layer_kernels(model, out_dir: Path = FIG_DIR):
    w = model.block1.block[0].weight.detach().cpu()  # (C_out, 3, 3, 3)
    n = min(32, w.shape[0])
    cols, rows = 8, int(np.ceil(n / 8))
    fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.5 * rows))
    for ax in np.array(axes).ravel():
        ax.axis("off")
    for i, ax in enumerate(np.array(axes).ravel()[:n]):
        k = w[i].permute(1, 2, 0).numpy()
        k = (k - k.min()) / (k.max() - k.min() + 1e-8)
        ax.imshow(k)
        ax.axis("off")
    fig.suptitle("第一层卷积核可视化（3×3×3，共 32 个）", fontsize=11)
    fig.tight_layout()
    return _save(fig, "13_kernels.png", out_dir)


@torch.no_grad()
def plot_feature_maps(model, paths, labels, transform, out_dir: Path = FIG_DIR):
    """对一张猫、一张狗，逐块展示前 8 个通道的特征图。"""
    picks = []
    for cls in (0, 1):
        idx = next(i for i, y in enumerate(labels) if y == cls)
        picks.append(idx)

    model.eval()
    blocks = [("block1", model.block1), ("block2", model.block2),
              ("block3", model.block3), ("block4", model.block4)]
    fig, axes = plt.subplots(2 * len(blocks) + 2, 8, figsize=(1.35 * 8, 1.35 * (2 * len(blocks) + 2)))
    for ax in axes.ravel():
        ax.axis("off")

    r = 0
    for i in picks:
        with Image.open(paths[i]) as im:
            img = im.convert("RGB")
            x = transform(img).unsqueeze(0).to(DEVICE)
            axes[r, 0].imshow(img)
            axes[r, 0].set_title(f"输入：{CLASS_NAMES_CN[CLASSES[labels[i]]]}", fontsize=8)
            axes[r, 0].axis("off")
            r += 1

        feat = x
        for name, blk in blocks:
            feat = blk(feat)
            fm = feat[0].cpu()
            for c in range(8):
                if c >= fm.shape[0]:
                    continue
                axes[r, c].imshow(fm[c].numpy(), cmap="viridis")
                axes[r, c].axis("off")
                if c == 0:
                    axes[r, c].set_ylabel(name, fontsize=7)
            axes[r, 0].set_title(f"{name}: 1×{fm.shape[0]}×{fm.shape[1]}×{fm.shape[2]}", fontsize=7.5)
            r += 1
    fig.suptitle("卷积特征图逐层演化（每行取前 8 个通道）", fontsize=11)
    fig.tight_layout()
    return _save(fig, "14_feature_maps.png", out_dir)
