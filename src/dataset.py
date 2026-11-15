"""数据读取与划分。

数据集为平铺的 100 张 JPG（cat.N.jpg / dog.N.jpg），文件名前缀即类别标签。
本模块负责：扫描样本 → 分层划分 → 构建 Dataset 与数据增强。
"""
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from config import CLASSES, DATA_RAW, IMG_SIZE


# ---------------- 样本扫描 ----------------
def collect_samples(root: Path = DATA_RAW) -> Tuple[List[str], List[int]]:
    """扫描目录，返回图片路径列表与对应标签（cat=0, dog=1）。"""
    paths, labels = [], []
    for cls_idx, cls in enumerate(CLASSES):
        for p in sorted(Path(root).glob(f"{cls}.*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                paths.append(str(p))
                labels.append(cls_idx)
    if not paths:
        raise FileNotFoundError(f"未在 {root} 找到图片，请先运行 src/prepare_data.py 解压数据集")
    return paths, labels


def dataset_summary(paths: Sequence[str], labels: Sequence[int]) -> dict:
    """统计类别数量与图像尺寸分布，写入实验日志。"""
    from collections import Counter

    size_counter = Counter()
    for p in paths:
        with Image.open(p) as im:
            size_counter[im.size] += 1
    return {
        "total": len(paths),
        "per_class": {c: int(sum(1 for y in labels if y == i)) for i, c in enumerate(CLASSES)},
        "unique_sizes": len(size_counter),
        "top_sizes": [
            {"size": list(s), "count": n} for s, n in size_counter.most_common(5)
        ],
    }


# ---------------- 划分 ----------------
def split_holdout(paths, labels, test_ratio: float, seed: int):
    """分层留出法：返回 (train_paths, train_labels, test_paths, test_labels)。"""
    idx = np.arange(len(paths))
    tr, te = train_test_split(idx, test_size=test_ratio, stratify=np.array(labels), random_state=seed)
    return (
        [paths[i] for i in tr], [labels[i] for i in tr],
        [paths[i] for i in te], [labels[i] for i in te],
    )


def stratified_folds(labels: Sequence[int], n_folds: int, seed: int):
    """分层 K 折：返回 [(train_idx, val_idx), ...]。"""
    idx = np.arange(len(labels))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(idx, np.array(labels))]


# ---------------- 数据增强 ----------------
def build_transforms(img_size: int = IMG_SIZE, train: bool = True, mean=None, std=None,
                     augment: bool = True):
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    if train and augment:
        ops = [
            # 小样本必须强增强：随机裁剪缩放 + 翻转 + 旋转 + 颜色扰动
            transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(12)], p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        ]
    else:
        ops = [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def compute_norm_stats(paths: Sequence[str], img_size: int = IMG_SIZE):
    """只用训练集统计像素均值/方差，避免信息从验证集泄漏。"""
    tf = transforms.Compose([transforms.Resize(int(img_size * 1.14)),
                             transforms.CenterCrop(img_size),
                             transforms.ToTensor()])
    total = 0
    s = torch.zeros(3, dtype=torch.float64)
    ss = torch.zeros(3, dtype=torch.float64)
    for p in paths:
        with Image.open(p) as im:
            x = tf(im.convert("RGB")).double()
        s += x.sum(dim=(1, 2))
        ss += (x ** 2).sum(dim=(1, 2))
        total += x.shape[1] * x.shape[2]
    mean = (s / total).tolist()
    var = (ss / total - (s / total) ** 2).clamp_min(1e-8)
    std = var.sqrt().tolist()
    return [round(v, 4) for v in mean], [round(v, 4) for v in std]


# ---------------- Dataset ----------------
class PetDataset(Dataset):
    def __init__(self, paths: Sequence[str], labels: Sequence[int], transform=None):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
            x = self.transform(img) if self.transform else transforms.ToTensor()(img)
        return x, self.labels[i]
