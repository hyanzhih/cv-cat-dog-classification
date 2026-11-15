"""训练流程：分层 K 折交叉验证 + 最终模型训练。

对外接口：
  fit(...)          —— 单次训练（含早停），返回历史曲线与最佳权重
  run_cross_validation(...) —— 5 折交叉验证，返回每折指标与折外预测（OOF）
  train_final_model(...)    —— 用训练集训练最终模型并在测试集上评估
"""
import copy
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (BATCH_SIZE, DEVICE, DROPOUT, EPOCHS, EPOCHS_FINAL, IMG_SIZE,
                    LABEL_SMOOTHING, LR, MIN_EPOCHS, PATIENCE, SELECT_BY, SWA_START_FRAC,
                    USE_SWA, USE_TTA, WEIGHT_DECAY)
from dataset import PetDataset, build_transforms, stratified_folds
from model import CatDogCNN
from utils import CsvLogger, binary_metrics


def make_loader(paths, labels, transform, shuffle: bool, batch_size: int = BATCH_SIZE):
    ds = PetDataset(paths, labels, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=(DEVICE.type == "cuda"), drop_last=False)


def _run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            count += x.size(0)
    return total_loss / count, correct / count


@torch.no_grad()
def predict_probs(model, loader, tta: bool = USE_TTA) -> (np.ndarray, np.ndarray):
    """返回 (标签, 属于 dog 类的概率)。tta=True 时对原图与水平翻转图取平均。"""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        prob = torch.softmax(model(x), dim=1)
        if tta:
            prob = 0.5 * (prob + torch.softmax(model(torch.flip(x, dims=[3])), dim=1))
        ys.append(y.numpy())
        ps.append(prob[:, 1].detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def fit(train_paths, train_labels, val_paths, val_labels, *, epochs: int,
        lr: float = LR, tag: str = "run", log_dir=None, mean=None, std=None,
        use_swa: bool = USE_SWA, min_epochs: int = MIN_EPOCHS,
        select_by: str = SELECT_BY, augment: bool = True, width: int = 32,
        use_bn: bool = True, dropout: float = DROPOUT, img_size: int = None,
        verbose: bool = True):
    """训练一个模型。

    - 验证集仅 20 张，直接用单轮准确率选模噪声极大，因此：
      * min_epochs 之前不做模型选择，避免「第 1 轮偶然全对」被当成最优；
      * 末段对权重做滑动平均（SWA），平均掉随机波动，泛化更稳；
      * 最后在「最佳单轮权重」与「SWA 权重」中按验证准确率取优。
    """
    img_size = img_size or IMG_SIZE
    tf_tr = build_transforms(img_size, train=True, mean=mean, std=std, augment=augment)
    tf_va = build_transforms(img_size, train=False, mean=mean, std=std)
    tl = make_loader(train_paths, train_labels, tf_tr, shuffle=True)
    vl = make_loader(val_paths, val_labels, tf_va, shuffle=False)

    model = CatDogCNN(dropout=dropout, width=width, use_bn=use_bn).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger = CsvLogger(log_dir / f"{tag}_history.csv") if log_dir else None
    hist = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_score, best_state, best_epoch, wait = -1.0, None, 0, 0
    swa_state, swa_n = None, 0
    swa_start = max(int(epochs * SWA_START_FRAC), min_epochs)
    t0 = time.time()

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, tl, criterion, optimizer)
        va_loss, va_acc = _run_epoch(model, vl, criterion)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        for k, v in zip(hist, (ep, tr_loss, tr_acc, va_loss, va_acc)):
            hist[k].append(v)
        if logger:
            logger.log(ep, tr_loss, tr_acc, va_loss, va_acc, cur_lr)

        # 滑动平均：从 swa_start 起对权重与 BN 统计量做累计平均
        if use_swa and ep >= swa_start:
            sd = model.state_dict()
            if swa_state is None:
                swa_state = {k: v.detach().clone().float() for k, v in sd.items()}
                swa_n = 1
            else:
                swa_n += 1
                for k, v in sd.items():
                    if v.dtype.is_floating_point:
                        swa_state[k].mul_(1 - 1 / swa_n).add_(v.detach().float(), alpha=1 / swa_n)

        if ep >= min_epochs:
            score = -va_loss if select_by == "loss" else va_acc
            if score > best_score + 1e-9:
                best_score, best_epoch = score, ep
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1

        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"  [{tag}] epoch {ep:3d}/{epochs} | train {tr_loss:.4f}/{tr_acc:.3f} "
                  f"| val {va_loss:.4f}/{va_acc:.3f} | lr {cur_lr:.2e}")

        if wait >= PATIENCE:
            if verbose:
                print(f"  [{tag}] 早停于 epoch {ep}")
            break

    # ---- 收尾：SWA 权重优先（无条件使用，不参与验证集挑拣，避免二次过拟合验证集） ----
    if use_swa and swa_state is not None:
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in swa_state.items()})
        chosen = f"swa(last{swa_n}epochs)"
    else:
        if best_state is not None:
            model.load_state_dict(best_state)
        chosen = "best"
    _, chosen_acc = _run_epoch(model, vl, criterion)

    hist["best_epoch"] = best_epoch
    hist["best_val_acc"] = chosen_acc
    hist["chosen"] = chosen
    hist["time_sec"] = round(time.time() - t0, 2)
    return model, hist


def run_cross_validation(paths, labels, *, n_folds: int, epochs: int, seed: int,
                         log_dir, mean=None, std=None, verbose: bool = True,
                         lr: float = LR, use_swa: bool = USE_SWA,
                         min_epochs: int = MIN_EPOCHS, select_by: str = SELECT_BY,
                         tta: bool = USE_TTA, augment: bool = True, width: int = 32,
                         use_bn: bool = True, dropout: float = DROPOUT,
                         img_size: int = None) -> Dict:
    """分层 K 折交叉验证；每折的验证集预测拼起来即为折外预测（OOF）。"""
    folds = stratified_folds(labels, n_folds, seed)
    labels_arr = np.array(labels)
    oof_prob = np.zeros(len(paths))
    oof_idx = np.zeros(len(paths), dtype=bool)
    per_fold = []
    histories = []
    fold_preds = []   # [(验证集标签, 验证集概率), ...] 用于逐折 ROC

    for k, (tr_idx, va_idx) in enumerate(folds, start=1):
        tr_paths = [paths[i] for i in tr_idx]
        tr_labels = [labels[i] for i in tr_idx]
        va_paths = [paths[i] for i in va_idx]
        va_labels = [labels[i] for i in va_idx]
        if verbose:
            print(f"第 {k}/{n_folds} 折：训练 {len(tr_paths)} 张 / 验证 {len(va_paths)} 张")

        model, hist = fit(tr_paths, tr_labels, va_paths, va_labels, epochs=epochs,
                          tag=f"fold{k}", log_dir=log_dir, mean=mean, std=std, verbose=verbose,
                          lr=lr, use_swa=use_swa, min_epochs=min_epochs, select_by=select_by,
                          augment=augment, width=width, use_bn=use_bn, dropout=dropout,
                          img_size=img_size)

        tf_va = build_transforms(img_size or IMG_SIZE, train=False, mean=mean, std=std)
        vl = make_loader(va_paths, va_labels, tf_va, shuffle=False)
        y, p = predict_probs(model, vl, tta=tta)
        m = binary_metrics(y, p)
        m.update({"fold": k, "best_epoch": hist["best_epoch"], "chosen": hist["chosen"],
                  "train_size": len(tr_paths), "val_size": len(va_paths)})
        per_fold.append(m)

        oof_prob[va_idx] = p
        oof_idx[va_idx] = True
        histories.append(hist)
        fold_preds.append((y, p))
        if verbose:
            print(f"  -> 该折验证准确率 {m['accuracy'] * 100:.2f}% | AUC {m['auc']:.4f} "
                  f"| 最佳 epoch {hist['best_epoch']}")

    assert oof_idx.all()
    accs = [m["accuracy"] for m in per_fold]
    summary = {
        "n_folds": n_folds,
        "fold_acc": accs,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs, ddof=1)),
        "f1_mean": float(np.mean([m["f1"] for m in per_fold])),
        "auc_mean": float(np.mean([m["auc"] for m in per_fold])),
        "oof": binary_metrics(labels_arr, oof_prob),
    }
    return {"per_fold": per_fold, "summary": summary, "oof_prob": oof_prob,
            "labels": labels_arr, "histories": histories, "fold_preds": fold_preds}


def train_final_model(train_paths, train_labels, test_paths, test_labels, *, epochs: int,
                      seed: int, log_dir, mean=None, std=None, img_size: int = None,
                      verbose: bool = True):
    """最终模型：训练集内再切 10% 作为早停用验证集，测试集只用于最后评估。"""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(train_paths))
    tr_i, va_i = train_test_split(idx, test_size=0.1, stratify=np.array(train_labels),
                                  random_state=seed)
    model, hist = fit([train_paths[i] for i in tr_i], [train_labels[i] for i in tr_i],
                      [train_paths[i] for i in va_i], [train_labels[i] for i in va_i],
                      epochs=epochs, tag="final", log_dir=log_dir, mean=mean, std=std,
                      img_size=img_size, verbose=verbose)
    tf_te = build_transforms(img_size or IMG_SIZE, train=False, mean=mean, std=std)
    te_loader = make_loader(test_paths, test_labels, tf_te, shuffle=False)
    y, p = predict_probs(model, te_loader)
    metrics = binary_metrics(y, p)
    metrics.update({"test_size": len(test_paths), "best_epoch": hist["best_epoch"],
                    "train_size": len(tr_i), "val_size": len(va_i)})
    return model, hist, metrics, (y, p)
