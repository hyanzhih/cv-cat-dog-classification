"""消融实验：逐项去掉关键设计，观察 5 折交叉验证准确率的变化。

运行：python src/ablation.py
输出：outputs/ablation.json、outputs/ablation.csv、outputs/figures/17_ablation.png
"""
import time

import numpy as np

import visualize as vis
from config import EPOCHS, IMG_SIZE, LOG_DIR, N_FOLDS, OUT_DIR, SEED
from dataset import collect_samples, compute_norm_stats, split_holdout
from train import run_cross_validation
from utils import save_json, set_seed, setup_stdout

setup_stdout()


def variants():
    """(名称, run_cross_validation 的关键字覆盖)"""
    return [
        ("完整模型（本文方法）", {}),
        ("无数据增强", {"augment": False}),
        ("无 BatchNorm", {"use_bn": False}),
        ("无 Dropout", {"dropout": 0.0}),
        ("输入 64×64", {"img_size": 64}),
        ("无 TTA 推理", {"tta": False}),
        ("网络宽度减半", {"width": 16}),
    ]


def main():
    paths, labels = collect_samples()
    tr_p, _, _, _ = split_holdout(paths, labels, 0.2, SEED)
    mean128, std128 = compute_norm_stats(tr_p, IMG_SIZE)
    mean64, std64 = compute_norm_stats(tr_p, 64)

    rows = []
    abl_log = LOG_DIR / "ablation"
    abl_log.mkdir(parents=True, exist_ok=True)
    for i, (name, over) in enumerate(variants(), start=1):
        img_size = over.get("img_size", IMG_SIZE)
        mean, std = (mean64, std64) if img_size == 64 else (mean128, std128)
        set_seed(SEED)
        t0 = time.time()
        print(f"[{i}/{len(variants())}] {name} ...", flush=True)
        cv = run_cross_validation(paths, labels, n_folds=N_FOLDS, epochs=EPOCHS, seed=SEED,
                                  log_dir=abl_log, mean=mean, std=std, verbose=False, **over)
        s = cv["summary"]
        rows.append({
            "name": name,
            "acc_mean": round(s["acc_mean"], 4),
            "acc_std": round(s["acc_std"], 4),
            "f1_mean": round(s["f1_mean"], 4),
            "auc_mean": round(s["auc_mean"], 4),
            "oof_acc": round(s["oof"]["accuracy"], 4),
            "fold_acc": [round(a, 4) for a in s["fold_acc"]],
            "sec": round(time.time() - t0, 1),
        })
        print(f"    acc {s['acc_mean']*100:.2f}% ± {s['acc_std']*100:.2f}% | "
              f"F1 {s['f1_mean']:.3f} | AUC {s['auc_mean']:.3f}", flush=True)

    save_json({"n_folds": N_FOLDS, "epochs": EPOCHS, "rows": rows}, OUT_DIR / "ablation.json")
    with open(OUT_DIR / "ablation.csv", "w", encoding="utf-8-sig") as f:
        f.write("变体,准确率(%),标准差(%),F1,AUC\n")
        for r in rows:
            f.write(f"{r['name']},{r['acc_mean']*100:.2f},{r['acc_std']*100:.2f},"
                    f"{r['f1_mean']:.4f},{r['auc_mean']:.4f}\n")

    vis.plot_ablation(rows)
    print("\n消融结果表：")
    for r in rows:
        print(f"  {r['name']:18s} {r['acc_mean']*100:5.2f}% ± {r['acc_std']*100:4.2f}%  "
              f"F1 {r['f1_mean']:.3f}  AUC {r['auc_mean']:.3f}")


if __name__ == "__main__":
    main()
