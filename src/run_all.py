"""一键复现整条实验流水线。

执行顺序：
  1) 解压数据并统计
  2) 数据探索可视化
  3) 5 折分层交叉验证（主结果）
  4) 留出法训练最终模型（保存权重，供推理使用）
  5) 评估与可解释性可视化
  6) 汇总所有指标到 outputs/metrics.json

用法：  python src/run_all.py
"""
import time

import numpy as np

import visualize as vis
from config import (BATCH_SIZE, DEVICE, EPOCHS, EPOCHS_FINAL, FIG_DIR, LABEL_SMOOTHING, LOG_DIR,
                    LR, MODEL_DIR, N_FOLDS, OUT_DIR, ROOT, SEED, TEST_RATIO, IMG_SIZE,
                    WEIGHT_DECAY)
from dataset import collect_samples, compute_norm_stats, split_holdout, build_transforms
from model import count_parameters
from prepare_data import prepare
from train import run_cross_validation, train_final_model
from utils import binary_metrics, fmt_pct, save_json, set_seed, setup_stdout

setup_stdout()


def main():
    t_start = time.time()
    print("=" * 68)
    print(f"猫狗图像二分类实验 · 自建 CNN · 输入 {IMG_SIZE}×{IMG_SIZE} · 5 折交叉验证")
    print("=" * 68)

    # ---------- 1. 数据 ----------
    print("\n[1/6] 准备数据")
    summary = prepare()
    paths, labels = collect_samples()

    print("\n[2/6] 数据探索可视化")
    vis.plot_class_distribution(summary)
    vis.plot_size_distribution(paths)
    vis.plot_samples(paths, labels)
    tr_p, tr_y, te_p, te_y = split_holdout(paths, labels, TEST_RATIO, SEED)
    mean, std = compute_norm_stats(tr_p, IMG_SIZE)
    print(f"  训练集像素统计：均值 {mean} 标准差 {std}")
    vis.plot_augmentation(tr_p[0], tr_y[0], build_transforms(train=True, mean=mean, std=std))

    # ---------- 2. 交叉验证 ----------
    print(f"\n[3/6] {N_FOLDS} 折分层交叉验证（主结果）")
    set_seed(SEED)
    cv = run_cross_validation(paths, labels, n_folds=N_FOLDS, epochs=EPOCHS, seed=SEED,
                              log_dir=LOG_DIR, mean=mean, std=std, verbose=True)

    s = cv["summary"]
    print(f"  平均验证准确率 {fmt_pct(s['acc_mean'])} ± {fmt_pct(s['acc_std'])}"
          f" | 平均 F1 {s['f1_mean']:.4f} | 平均 AUC {s['auc_mean']:.4f}")
    print(f"  折外(OOF)整体准确率 {fmt_pct(s['oof']['accuracy'])} | AUC {s['oof']['auc']:.4f}")

    print("\n[4/6] 交叉验证结果可视化")
    vis.plot_cv_curves(cv["histories"])
    vis.plot_fold_accuracy(cv["per_fold"])
    vis.plot_confusion(s["oof"]["confusion"], f"折外预测混淆矩阵（准确率 {s['oof']['accuracy']*100:.1f}%）",
                       "08_confusion_oof.png")
    curves = [(y, p, f"第{i}折") for i, (y, p) in enumerate(cv["fold_preds"], start=1)]
    vis.plot_roc(curves, name="09_roc_folds.png")
    vis.plot_roc([(cv["labels"], cv["oof_prob"], "折外预测")], name="09b_roc_oof.png")
    vis.plot_prob_hist(cv["labels"], cv["oof_prob"], "折外预测概率分布", "11_prob_hist.png")
    vis.plot_misclassified(paths, labels, cv["oof_prob"])

    # ---------- 3. 最终模型 ----------
    print("\n[5/6] 留出法训练最终模型")
    model, hist, metrics, (y_te, p_te) = train_final_model(
        tr_p, tr_y, te_p, te_y, epochs=EPOCHS_FINAL, seed=SEED, log_dir=LOG_DIR,
        mean=mean, std=std, verbose=True)
    print(f"  测试集准确率 {fmt_pct(metrics['accuracy'])} | 精确率 {metrics['precision']:.3f} "
          f"| 召回率 {metrics['recall']:.3f} | F1 {metrics['f1']:.3f} | AUC {metrics['auc']:.3f}")

    import torch
    ckpt = MODEL_DIR / "catdog_cnn.pth"
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                "img_size": IMG_SIZE, "classes": ["cat", "dog"],
                "test_metrics": metrics}, ckpt)
    print(f"  权重已保存：{ckpt.relative_to(ROOT)}")

    vis.plot_final_curve(hist)
    vis.plot_confusion(metrics["confusion"], f"测试集混淆矩阵（准确率 {metrics['accuracy']*100:.1f}%）",
                       "15_confusion_test.png")
    vis.plot_roc([(cv["labels"], cv["oof_prob"], "5 折折外预测"),
                  (y_te, p_te, "留出测试集")], name="16_roc_compare.png")

    print("\n[6/6] 模型可解释性可视化")
    vis.plot_first_layer_kernels(model)
    vis.plot_feature_maps(model, te_p, te_y, build_transforms(train=False, mean=mean, std=std))

    # ---------- 汇总 ----------
    result = {
        "config": {
            "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
            "epochs_cv": EPOCHS, "epochs_final": EPOCHS_FINAL, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "label_smoothing": LABEL_SMOOTHING,
            "n_folds": N_FOLDS, "test_ratio": TEST_RATIO, "seed": SEED,
            "device": str(DEVICE), "params": count_parameters(model),
        },
        "dataset": summary,
        "norm_stats": {"mean": mean, "std": std},
        "cross_validation": {"per_fold": cv["per_fold"], "summary": s},
        "final_model": {"test_metrics": metrics, "history_best_epoch": hist["best_epoch"],
                        "history_best_val_acc": hist["best_val_acc"],
                        "train_time_sec": hist["time_sec"]},
        "predictions": {
            "oof_prob": [round(float(v), 4) for v in cv["oof_prob"]],
            "oof_label": [int(v) for v in cv["labels"]],
            "test_prob": [round(float(v), 4) for v in p_te],
            "test_label": [int(v) for v in y_te],
        },
        "figures": sorted(p.name for p in FIG_DIR.glob("*.png")),
        "total_time_sec": round(time.time() - t_start, 1),
    }
    save_json(result, OUT_DIR / "metrics.json")
    print(f"\n全部完成，耗时 {result['total_time_sec']} s。指标写入 outputs/metrics.json")


if __name__ == "__main__":
    main()
