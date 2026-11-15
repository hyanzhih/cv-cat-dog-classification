"""实验全局配置：路径、超参数、随机种子。

所有脚本共享同一份配置，保证实验可复现。
"""
from pathlib import Path
import torch

# ---------------- 路径 ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
LOG_DIR = OUT_DIR / "logs"
for _d in (DATA_RAW, OUT_DIR, FIG_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------- 数据 ----------------
CLASSES = ("cat", "dog")
CLASS_NAMES_CN = {"cat": "猫", "dog": "狗"}
IMG_SIZE = 128            # 输入分辨率（原始图像尺寸不一，统一缩放）
TEST_RATIO = 0.2          # 留出法测试集比例（仅用于最终模型）
N_FOLDS = 5               # 分层 K 折交叉验证折数

# ---------------- 训练 ----------------
SEED = 42
BATCH_SIZE = 16
EPOCHS = 100              # 交叉验证每折训练轮数
EPOCHS_FINAL = 120        # 最终模型训练轮数
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
PATIENCE = 25             # 早停耐心值（验证准确率不再提升的轮数）
DROPOUT = 0.5
MIN_EPOCHS = 20           # 少于此轮数不参与选模（验证集太小，避免早轮偶然最优）
SELECT_BY = "acc"         # 选模准则：acc（验证准确率）/ loss（验证损失）
USE_SWA = False           # 权重滑动平均：早停截断下平均窗口过短，实测不稳定，默认关闭
SWA_START_FRAC = 0.6      # 从总轮数的 60% 处开始平均
USE_TTA = True            # 推理时水平翻转增强（TTA）

# ---------------- 设备 ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- 绘图风格 ----------------
C_MAIN = "#2F5597"        # 主色（深蓝）
C_ACCENT = "#C00000"      # 强调色（红）
C_GRAY = "#7F7F7F"
