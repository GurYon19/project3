# part3/inference.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

from part3.model import FixedSlotDetector


VOC_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (30, 144, 255),
    (255, 165, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_classes(classes_json: str | Path) -> List[str]:
    import json
    classes_json = Path(classes_json)
    obj = json.loads(classes_json.read_text(encoding="utf-8"))
    return obj["classes"]


def load_model(ckpt_path: str | Path, classes: List[str], image_size: int, max_objects: int, device: torch.device):
    num_classes = len(classes)
    model = FixedSlotDetector(
        num_classes=num_classes,
        max_objects=max_objects,
        pretrained=False,   # weights come from checkpoint
        freeze_backbone=False,
        image_size=image_size,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_on_tensor(model, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    x: [1,3,H,W] float in [0,1]
    Returns:
      boxes: [K,4] xyxy in pixel coords (on resized image)
      cls:   [K] predicted class id (0..C for bg)
      conf:  [K] confidence (max softmax prob)
    """
    out = model(x)
    boxes = out["boxes"][0]          # [K,4]
    logits = out["logits"][0]        # [K,C+1]
    probs = F.softmax(logits, dim=-1)
    conf, cls = probs.max(dim=-1)    # [K]
    return boxes.cpu().numpy(), cls.cpu().numpy(), conf.cpu().numpy()


def draw_pil(
    img: Image.Image,
    boxes: np.ndarray,
    cls: np.ndarray,
    conf: np.ndarray,
    class_names: List[str],
    bg_id: int,
    conf_thresh: float,
    topk: int = 3,
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # Try to load a default font
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    # rank by confidence (excluding bg)
    keep = [(i, float(conf[i])) for i in range(len(conf)) if int(cls[i]) != bg_id and float(conf[i]) >= conf_thresh]
    keep.sort(key=lambda x: x[1], reverse=True)
    keep = keep[:topk]

    for rank, (i, c) in enumerate(keep):
        x1, y1, x2, y2 = boxes[i].tolist()
        x1 = max(0, min(W - 1, int(round(x1))))
        y1 = max(0, min(H - 1, int(round(y1))))
        x2 = max(0, min(W - 1, int(round(x2))))
        y2 = max(0, min(H - 1, int(round(y2))))

        color = VOC_COLORS[rank % len(VOC_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        name = class_names[int(cls[i])]
        text = f"{name} {c:.2f}"

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, max(0, y1 - th - 2)), text, fill=(0, 0, 0), font=font)

    return img


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    # [3,H,W] float in [0,1]
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def run_image(args):
    device = pick_device()
    classes = load_classes(args.classes_json)
    bg_id = len(classes)

    model = load_model(args.checkpoint, classes, args.image_size, args.max_objects, device)

    img0 = Image.open(args.image).convert("RGB")
    img = img0.resize((args.image_size, args.image_size), resample=Image.BILINEAR)
    x = pil_to_tensor(img).unsqueeze(0).to(device)

    boxes, cls, conf = predict_on_tensor(model, x)

    vis = draw_pil(img.copy(), boxes, cls, conf, classes, bg_id, args.conf_thresh, topk=args.topk)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.image).stem}_pred.jpg"
    vis.save(out_path)
    print(f"[OK] saved image: {out_path}")


def run_video(args):
    device = pick_device()
    classes = load_classes(args.classes_json)
    bg_id = len(classes)

    model = load_model(args.checkpoint, classes, args.image_size, args.max_objects, device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.video).stem}_pred.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (args.image_size, args.image_size))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb).resize((args.image_size, args.image_size), resample=Image.BILINEAR)

        x = pil_to_tensor(pil).unsqueeze(0).to(device)
        boxes, cls, conf = predict_on_tensor(model, x)
        vis = draw_pil(pil.copy(), boxes, cls, conf, classes, bg_id, args.conf_thresh, topk=args.topk)

        vis_bgr = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        writer.write(vis_bgr)

        frame_idx += 1
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break
        if frame_idx % 50 == 0:
            print(f"[VIDEO] processed {frame_idx} frames...")

    cap.release()
    writer.release()
    print(f"[OK] saved video: {out_path}")


def parse_args():
    p = argparse.ArgumentParser("Part3 inference (fixed-slot detector)")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoints/part3/best.pth")
    p.add_argument("--classes-json", type=str, default="datasets/part3/classes.json")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--max-objects", type=int, default=3)
    p.add_argument("--conf-thresh", type=float, default=0.35)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--out-dir", type=str, default="outputs/part3")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str, help="Path to image")
    g.add_argument("--video", type=str, help="Path to video")

    p.add_argument("--max-frames", type=int, default=-1, help="For video: limit frames (default: no limit)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.image:
        run_image(args)
    else:
        run_video(args)


if __name__ == "__main__":
    main()
