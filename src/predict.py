"""推理脚本：对单张图片或整个文件夹做猫狗预测。

用法：
  python src/predict.py --image data/raw/cat.1.jpg
  python src/predict.py --dir data/raw --topk 10
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

from config import CLASSES, CLASS_NAMES_CN, DEVICE, MODEL_DIR, ROOT
from dataset import build_transforms
from model import CatDogCNN
from utils import setup_stdout


def load_model(ckpt_path: Path = MODEL_DIR / "catdog_cnn.pth"):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = CatDogCNN().to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tf = build_transforms(ckpt.get("img_size", 128), train=False,
                          mean=ckpt.get("mean"), std=ckpt.get("std"))
    return model, tf, ckpt


@torch.no_grad()
def predict_image(model, tf, image_path: str):
    with Image.open(image_path) as im:
        x = tf(im.convert("RGB")).unsqueeze(0).to(DEVICE)
    prob = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    idx = int(prob.argmax())
    return CLASSES[idx], float(prob[idx]), prob


def main():
    setup_stdout()
    ap = argparse.ArgumentParser(description="猫狗图像分类推理")
    ap.add_argument("--image", type=str, help="单张图片路径")
    ap.add_argument("--dir", type=str, help="图片文件夹路径")
    ap.add_argument("--ckpt", type=str, default=str(MODEL_DIR / "catdog_cnn.pth"))
    args = ap.parse_args()

    model, tf, ckpt = load_model(Path(args.ckpt))
    print(f"已加载权重：{Path(args.ckpt).name}（测试集准确率 "
          f"{ckpt.get('test_metrics', {}).get('accuracy', float('nan')) * 100:.1f}%）")

    if args.image:
        cls, conf, prob = predict_image(model, tf, args.image)
        print(f"{Path(args.image).name} -> {CLASS_NAMES_CN[cls]}（{cls}），置信度 {conf * 100:.2f}%")
        print(f"  概率：猫 {prob[0] * 100:.2f}% | 狗 {prob[1] * 100:.2f}%")

    if args.dir:
        files = sorted(p for p in Path(args.dir).glob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        correct = total = 0
        print(f"\n批量预测 {len(files)} 张图片：")
        for p in files:
            cls, conf, _ = predict_image(model, tf, str(p))
            guess = p.stem.split(".")[0]
            ok = guess in CLASSES and guess == cls
            if guess in CLASSES:
                total += 1
                correct += int(ok)
            print(f"  {'OK ' if ok else 'ERR'} {p.name:16s} -> {cls:4s} ({conf * 100:5.2f}%)")
        if total:
            print(f"\n按文件名真值统计：{correct}/{total} = {correct / total * 100:.2f}%")

    if not args.image and not args.dir:
        ap.print_help()


if __name__ == "__main__":
    main()
