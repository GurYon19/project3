# part3/run_inference_batch.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from part3.model import FixedSlotDetector, ModelConfig


VOC_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (30, 144, 255),
    (255, 165, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Batch inference (random sample of images from an index JSON)")
    p.add_argument("--index-json", type=str, default="datasets/part3/test.json")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--classes-json", type=str, default="datasets/part3/classes.json")
    p.add_argument("--out-dir", type=str, default="outputs/part3/infer_batch")

    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--conf-thresh", type=float, default=0.25)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--max-objects", type=int, default=3)

    # Must match how you trained (default in train.py)
    p.add_argument("--backbone", type=str, default="mobilenet_v3_small")
    p.add_argument("--pretrained", action="store_true")

    return p.parse_args()


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_classes(classes_json: str | Path) -> Tuple[List[str], int]:
    """
    Supports:
      1) {"classes":[...], "bg_id":3}
      2) {"classes":[...]} where bg_id = len(classes)-1 (train.py logic)
    """
    obj = json.loads(Path(classes_json).read_text(encoding="utf-8"))
    classes = obj["classes"]
    if "bg_id" in obj:
        bg_id = int(obj["bg_id"])
    else:
        bg_id = len(classes) - 1
    return classes, bg_id


def load_model(
    ckpt_path: str | Path,
    classes: List[str],
    bg_id: int,
    image_size: int,
    max_objects: int,
    device: torch.device,
    backbone: str,
    pretrained: bool,
) -> FixedSlotDetector:
    num_classes_total = len(classes)

    cfg = ModelConfig(
        image_size=int(image_size),
        max_objects=int(max_objects),
        num_classes_total=int(num_classes_total),
        bg_id=int(bg_id),
        backbone=str(backbone),
        pretrained=bool(pretrained),
    )
    model = FixedSlotDetector(cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    model.eval()
    return model


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0  # HWC float
    arr = np.transpose(arr, (2, 0, 1))              # CHW
    return torch.from_numpy(arr)


@torch.no_grad()
def predict(model: FixedSlotDetector, x: torch.Tensor):
    """
    Returns:
      boxes: [K,4] in resized image coords
      cls:   [K]
      conf:  [K]
    """
    out = model(x)
    boxes = out["boxes"][0]        # [K,4]
    logits = out["logits"][0]      # [K,C]
    probs = F.softmax(logits, dim=-1)
    conf, cls = probs.max(dim=-1)
    return boxes.detach().cpu().numpy(), cls.detach().cpu().numpy(), conf.detach().cpu().numpy()


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

        name = class_names[int(cls[i])] if 0 <= int(cls[i]) < len(class_names) else str(int(cls[i]))
        text = f"{name} {sc:.2f}"

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, max(0, y1 - th - 2)), text, fill=(0, 0, 0), font=font)

    return img


def main() -> None:
    args = parse_args()
    device = pick_device()

    classes, bg_id = load_classes(args.classes_json)

    data = json.loads(Path(args.index_json).read_text(encoding="utf-8"))
    paths = [s["image_path"] for s in data if "image_path" in s]

    random.seed(args.seed)
    pick = random.sample(paths, min(args.n, len(paths)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(
        ckpt_path=args.checkpoint,
        classes=classes,
        bg_id=bg_id,
        image_size=args.image_size,
        max_objects=args.max_objects,
        device=device,
        backbone=args.backbone,
        pretrained=args.pretrained,
    )

    print(f"[DEVICE] {device.type}")
    print(f"[MODEL] ckpt={args.checkpoint} backbone={args.backbone} pretrained={args.pretrained}")
    print(f"[CLASSES] n={len(classes)} bg_id={bg_id} classes={classes}")
    print(f"[BATCH] n={len(pick)} out_dir={out_dir}")
    print(f"[PRED] conf_thresh={args.conf_thresh} topk={args.topk}")

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