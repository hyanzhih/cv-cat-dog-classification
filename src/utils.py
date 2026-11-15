"""通用工具：随机种子、日志、指标计算、绘图样式。"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def setup_stdout():
    """Windows 控制台默认 GBK，统一改为 UTF-8 避免中文打印报错。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_plot_style():
    """中文字体 + 干净的图表样式。"""
    for f in ("C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"):
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
            name = font_manager.FontProperties(fname=f).get_name()
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
            break
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "font.size": 10,
    })


# ---------------- 指标 ----------------
def binary_metrics(y_true, y_prob, thr: float = 0.5):
    """二分类指标（正类 = dog）。y_prob 为属于正类的概率。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= thr).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    def _div(a, b):
        return float(a) / float(b) if b else 0.0

    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    f1 = _div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    acc = _div(tp + tn, len(y_true))
    spec = _div(tn, tn + fp)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": spec,
        "auc": roc_auc(y_true, y_prob),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def roc_auc(y_true, y_prob):
    """AUC = P(正样本得分 > 负样本得分)，用秩和公式计算，等价于 sklearn 结果。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_prob)
    ranks = np.empty(len(y_prob), dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)
    # 处理并列得分：取平均秩
    sorted_prob = y_prob[order]
    i = 0
    while i < len(sorted_prob):
        j = i
        while j + 1 < len(sorted_prob) and sorted_prob[j + 1] == sorted_prob[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    sum_rank_pos = ranks[y_true == 1].sum()
    return float((sum_rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve_points(y_true, y_prob):
    """返回 (fpr, tpr, thresholds)，均按 fpr 递增排列，可直接用于绘制与插值。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[order]
    prob_sorted = y_prob[order]
    P = max(int(y_sorted.sum()), 1)
    N = max(int(len(y_sorted) - y_sorted.sum()), 1)
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    fpr = np.r_[0.0, fp / N]
    tpr = np.r_[0.0, tp / P]
    thresholds = np.r_[np.inf, prob_sorted]
    return fpr, tpr, thresholds


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class CsvLogger:
    """把每个 epoch 的指标追加写入 csv，便于后续分析。"""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")

    def log(self, epoch, tr_loss, tr_acc, va_loss, va_acc, lr):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr_loss:.6f},{tr_acc:.6f},{va_loss:.6f},{va_acc:.6f},{lr:.8f}\n")


def fmt_pct(x, nd=2):
    return f"{x * 100:.{nd}f}%"
