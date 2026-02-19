# part3/run_inference_batch.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from part3.model import FixedSlotDetector


VOC_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (30, 144, 255),
    (255, 165, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def parse_args():
    p = argparse.ArgumentParser("Batch inference (in-process)")
    p.add_argument("--index-json", type=str, default="datasets/part3/test.json")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--classes-json", type=str, default="datasets/part3/classes.json")
    p.add_argument("--out-dir", type=str, default="outputs/part3/infer_batch")

    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--conf-thresh", type=float, default=0.35)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--max-objects", type=int, default=3)
    return p.parse_args()


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_classes(classes_json: str | Path) -> List[str]:
    obj = json.loads(Path(classes_json).read_text(encoding="utf-8"))
    return obj["classes"]


def load_model(ckpt_path: str | Path, classes: List[str], image_size: int, max_objects: int, device: torch.device):
    num_classes = len(classes)
    model = FixedSlotDetector(
        num_classes=num_classes,
        max_objects=max_objects,
        pretrained=False,
        freeze_backbone=False,
        image_size=image_size,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


@torch.no_grad()
def predict(model, x: torch.Tensor):
    out = model(x)
    boxes = out["boxes"][0]        # [K,4]
    logits = out["logits"][0]      # [K,C+1]
    probs = F.softmax(logits, dim=-1)
    conf, cls = probs.max(dim=-1)
    return boxes.cpu().numpy(), cls.cpu().numpy(), conf.cpu().numpy()


def draw_boxes(
    img: Image.Image,
    boxes: np.ndarray,
    cls: np.ndarray,
    conf: np.ndarray,
    class_names: List[str],
    bg_id: int,
    conf_thresh: float,
    topk: int,
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    W, H = img.size
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    keep = [(i, float(conf[i])) for i in range(len(conf)) if int(cls[i]) != bg_id and float(conf[i]) >= conf_thresh]
    keep.sort(key=lambda x: x[1], reverse=True)
    keep = keep[:topk]

    for rank, (i, sc) in enumerate(keep):
        x1, y1, x2, y2 = boxes[i].tolist()
        x1 = max(0, min(W - 1, int(round(x1))))
        y1 = max(0, min(H - 1, int(round(y1))))
        x2 = max(0, min(W - 1, int(round(x2))))
        y2 = max(0, min(H - 1, int(round(y2))))

        color = VOC_COLORS[rank % len(VOC_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        name = class_names[int(cls[i])]
        text = f"{name} {sc:.2f}"
        
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, max(0, y1 - th - 2)), text, fill=(0, 0, 0), font=font)

    return img


def main():
    args = parse_args()
    device = pick_device()

    classes = load_classes(args.classes_json)
    bg_id = len(classes)

    data = json.loads(Path(args.index_json).read_text(encoding="utf-8"))
    paths = [s["image_path"] for s in data]

    random.seed(args.seed)
    pick = random.sample(paths, min(args.n, len(paths)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, classes, args.image_size, args.max_objects, device)

    print(f"[DEVICE] {device}")
    print(f"[MODEL] ckpt={args.checkpoint}")
    print(f"[BATCH] n={len(pick)} out_dir={out_dir}")

    for i, p in enumerate(pick, 1):
        img0 = Image.open(p).convert("RGB")
        img = img0.resize((args.image_size, args.image_size), resample=Image.BILINEAR)

        x = pil_to_tensor(img).unsqueeze(0).to(device)
        boxes, cls, conf = predict(model, x)

        vis = draw_boxes(img.copy(), boxes, cls, conf, classes, bg_id, args.conf_thresh, args.topk)

        out_path = out_dir / f"{Path(p).stem}_pred.jpg"
        vis.save(out_path)
        print(f"[{i}/{len(pick)}] saved {out_path}")

    print("[OK] done")


if __name__ == "__main__":
    main()
