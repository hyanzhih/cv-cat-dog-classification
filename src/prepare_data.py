"""数据集解压与统计（可重复运行，已存在则跳过）。"""
import shutil
import zipfile

from config import DATA_RAW, ROOT
from dataset import collect_samples, dataset_summary
from utils import save_json, setup_stdout

ZIP_CANDIDATES = list(ROOT.glob("*.zip"))


def prepare():
    setup_stdout()
    existing = list(DATA_RAW.glob("*.jpg"))
    if existing:
        print(f"数据已就绪（{len(existing)} 张），跳过解压。")
    else:
        if not ZIP_CANDIDATES:
            raise FileNotFoundError("工作区根目录未找到数据集压缩包")
        src = ZIP_CANDIDATES[0]
        print(f"解压 {src.name} ...")
        with zipfile.ZipFile(src) as z:
            for name in z.namelist():
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    target = DATA_RAW / name.split("/")[-1]
                    if not target.exists():
                        with z.open(name) as f, open(target, "wb") as o:
                            shutil.copyfileobj(f, o)

    paths, labels = collect_samples()
    summary = dataset_summary(paths, labels)
    save_json(summary, ROOT / "outputs" / "logs" / "dataset_summary.json")
    print(f"样本总数 {summary['total']}，类别分布 {summary['per_class']}，不同尺寸数 {summary['unique_sizes']}")
    return summary


if __name__ == "__main__":
    prepare()
