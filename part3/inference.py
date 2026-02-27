# part3/inference.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

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


def load_model(
    ckpt_path: str | Path,
    classes: List[str],
    image_size: int,
    max_objects: int,
    device: torch.device,
):
    num_classes = len(classes)
    model = FixedSlotDetector(
        num_classes=num_classes,
        max_objects=max_objects,
        pretrained=False,  # weights come from checkpoint
        freeze_backbone=False,
        image_size=image_size,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def parse_class_thresh(s: str, class_names: List[str]) -> Dict[int, float]:
    """
    Parse "person=0.4,car=0.2,dog=0.35" -> {0:0.4, 1:0.2, 2:0.35}
    """
    if not s.strip():
        return {}
    out: Dict[int, float] = {}
    name_to_id = {n: i for i, n in enumerate(class_names)}
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Bad --class-thresh chunk: {chunk} (expected name=thr)")
        name, val = chunk.split("=", 1)
        name = name.strip()
        thr = float(val.strip())
        if name not in name_to_id:
            raise ValueError(f"Unknown class in --class-thresh: {name}. Known: {class_names}")
        out[name_to_id[name]] = thr
    return out


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)


def nms_classwise(
    boxes: np.ndarray,  # [N,4]
    scores: np.ndarray,  # [N]
    classes: np.ndarray,  # [N]
    iou_thresh: float,
) -> List[int]:
    """
    Simple class-wise NMS.
    Returns kept indices.
    """
    keep_all: List[int] = []
    for c in np.unique(classes):
        idxs = np.where(classes == c)[0]
        if idxs.size == 0:
            continue

        # sort by score desc
        idxs = idxs[np.argsort(scores[idxs])[::-1]]

        kept: List[int] = []
        while idxs.size > 0:
            i = int(idxs[0])
            kept.append(i)

            if idxs.size == 1:
                break

            rest = idxs[1:]
            sup = []
            for j in rest:
                if iou_xyxy(boxes[i], boxes[int(j)]) > iou_thresh:
                    sup.append(int(j))
            # keep those not suppressed
            idxs = np.array([j for j in rest if int(j) not in sup], dtype=np.int64)

        keep_all.extend(kept)

    # return all kept, sorted by score desc
    keep_all.sort(key=lambda i: float(scores[i]), reverse=True)
    return keep_all


@torch.no_grad()
def predict_on_tensor(
    model,
    x: torch.Tensor,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    x: [1,3,H,W] float in [0,1]

    Returns:
      boxes: [K,4] xyxy in pixel coords (on resized image)
      cls:   [K] predicted class id (0..C-1) (NO background)
      score: [K] confidence = max softmax prob over non-bg classes
    """
    out = model(x)
    boxes = out["boxes"][0]           # [K,4]
    logits = out["logits"][0]         # [K,C+1]
    probs = F.softmax(logits, dim=-1) # [K,C+1]

    # exclude background column for scoring
    p_nobg = probs[:, :num_classes]            # [K,C]
    score, cls = p_nobg.max(dim=-1)            # [K]
    return boxes.cpu().numpy(), cls.cpu().numpy(), score.cpu().numpy()


def draw_pil(
    img: Image.Image,
    boxes: np.ndarray,
    cls: np.ndarray,
    score: np.ndarray,
    class_names: List[str],
    conf_thresh: float,
    class_thresh: Dict[int, float],
    topk: int = 3,
    use_nms: bool = False,
    nms_iou: float = 0.5,
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    W, H = img.size

    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    # Apply per-class threshold (fallback to global)
    cand_idx = []
    for i in range(len(score)):
        c = int(cls[i])
        thr = float(class_thresh.get(c, conf_thresh))
        if float(score[i]) >= thr:
            cand_idx.append(i)

    if len(cand_idx) == 0:
        return img

    cand_idx = np.array(cand_idx, dtype=np.int64)

    c_boxes = boxes[cand_idx]
    c_cls = cls[cand_idx]
    c_score = score[cand_idx]

    # Optional NMS (class-wise)
    if use_nms:
        keep_rel = nms_classwise(c_boxes, c_score, c_cls, iou_thresh=float(nms_iou))
        cand_idx = cand_idx[keep_rel]
        c_boxes = boxes[cand_idx]
        c_cls = cls[cand_idx]
        c_score = score[cand_idx]

    # Sort by score and take topk
    order = np.argsort(c_score)[::-1]
    order = order[:topk]
    c_boxes = c_boxes[order]
    c_cls = c_cls[order]
    c_score = c_score[order]

    for rank in range(len(order)):
        x1, y1, x2, y2 = c_boxes[rank].tolist()
        x1 = max(0, min(W - 1, int(round(x1))))
        y1 = max(0, min(H - 1, int(round(y1))))
        x2 = max(0, min(W - 1, int(round(x2))))
        y2 = max(0, min(H - 1, int(round(y2))))

        color = VOC_COLORS[rank % len(VOC_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        name = class_names[int(c_cls[rank])]
        text = f"{name} {float(c_score[rank]):.2f}"

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        draw.rectangle([x1, max(0, y1 - th - 4), x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, max(0, y1 - th - 2)), text, fill=(0, 0, 0), font=font)

    return img


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def run_image(args):
    device = pick_device()
    classes = load_classes(args.classes_json)
    num_classes = len(classes)
    class_thresh = parse_class_thresh(args.class_thresh, classes)

    model = load_model(args.checkpoint, classes, args.image_size, args.max_objects, device)

    img0 = Image.open(args.image).convert("RGB")
    img = img0.resize((args.image_size, args.image_size), resample=Image.BILINEAR)
    x = pil_to_tensor(img).unsqueeze(0).to(device)

    boxes, cls, score = predict_on_tensor(model, x, num_classes=num_classes)

    vis = draw_pil(
        img.copy(),
        boxes,
        cls,
        score,
        classes,
        conf_thresh=args.conf_thresh,
        class_thresh=class_thresh,
        topk=args.topk,
        use_nms=args.use_nms,
        nms_iou=args.nms_iou,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(args.image).stem}_pred.jpg"
    vis.save(out_path)
    print(f"[OK] saved image: {out_path}")


def run_video(args):
    device = pick_device()
    classes = load_classes(args.classes_json)
    num_classes = len(classes)
    class_thresh = parse_class_thresh(args.class_thresh, classes)

    model = load_model(args.checkpoint, classes, args.image_size, args.max_objects, device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

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
        boxes, cls, score = predict_on_tensor(model, x, num_classes=num_classes)

        vis = draw_pil(
            pil.copy(),
            boxes,
            cls,
            score,
            classes,
            conf_thresh=args.conf_thresh,
            class_thresh=class_thresh,
            topk=args.topk,
            use_nms=args.use_nms,
            nms_iou=args.nms_iou,
        )

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
    p.add_argument(
        "--class-thresh",
        type=str,
        default="",
        help='Per-class thresholds like "person=0.4,car=0.2,dog=0.35". Overrides --conf-thresh for those classes.',
    )

    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--out-dir", type=str, default="outputs/part3")

    p.add_argument("--use-nms", action="store_true", help="Enable class-wise NMS in inference.")
    p.add_argument("--nms-iou", type=float, default=0.5, help="IoU threshold for NMS (default 0.5).")

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