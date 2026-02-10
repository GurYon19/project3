# part3/inference.py
"""
Part 3 inference:
- Loads a trained Part3Detector checkpoint
- Runs on a single image OR a folder of images
- Converts fixed K=3 slots -> final detections:
    score = sigmoid(obj_logit) * max_softmax(cls_logits)
- Applies score threshold + NMS
- Draws boxes + labels + scores
- Saves outputs to outputs/part3/inference/

Usage examples (from repo root):

# Run on a folder (recommended sanity check)
python -m part3.inference --checkpoint checkpoints/part3/best_model.pth --folder data/part3/test/images --max-images 50

# Run on a single image
python -m part3.inference --checkpoint checkpoints/part3/best_model.pth --image data/part3/test/images/<some.jpg>

Notes:
- Expects images are RGB photos (jpg/png).
- Uses config.PART3_CONFIG['class_names'] for label display.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch

import config
from part3.model import Part3Detector


# -------------------------
# Utilities
# -------------------------

def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Support multiple checkpoint formats
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            # might already be a state_dict-like dict
            state = ckpt
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)


def preprocess_image_bgr(img_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    """
    img_bgr: HxWx3 uint8
    returns: (1,3,image_size,image_size) float32 ImageNet normalized (RGB)
    """
    img_resized = cv2.resize(img_bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.array(config.IMAGENET_MEAN, dtype=np.float32)
    std = np.array(config.IMAGENET_STD, dtype=np.float32)
    img_rgb = (img_rgb - mean) / std

    t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).contiguous()  # (1,3,H,W)
    return t


def xyxy_norm_to_px(box: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = int(np.clip(x1 * w, 0, w - 1))
    x2 = int(np.clip(x2 * w, 0, w - 1))
    y1 = int(np.clip(y1 * h, 0, h - 1))
    y2 = int(np.clip(y2 * h, 0, h - 1))
    # enforce ordering
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    return x1, y1, x2, y2


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """a,b: (4,) xyxy in normalized coords"""
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
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return inter / union


def nms(
    boxes: np.ndarray, scores: np.ndarray, iou_thresh: float
) -> List[int]:
    """
    Classic NMS.
    boxes: (N,4) normalized xyxy
    scores: (N,)
    returns: indices kept
    """
    if len(boxes) == 0:
        return []

    idxs = scores.argsort()[::-1]
    keep: List[int] = []

    while idxs.size > 0:
        i = int(idxs[0])
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        suppressed = []
        for j in rest:
            if iou_xyxy(boxes[i], boxes[int(j)]) > iou_thresh:
                suppressed.append(int(j))
        idxs = np.array([int(j) for j in rest if int(j) not in suppressed], dtype=np.int64)

    return keep


def decode_slots(
    pred_boxes: torch.Tensor,      # (K,4)
    pred_obj_logits: torch.Tensor, # (K,)
    pred_cls_logits: torch.Tensor, # (K,C)
    score_thresh: float,
) -> Dict[str, np.ndarray]:
    """
    Returns dict with keys: boxes(N,4), scores(N,), labels(N,)
    """
    K = pred_boxes.shape[0]
    obj = torch.sigmoid(pred_obj_logits)                 # (K,)
    cls_prob = torch.softmax(pred_cls_logits, dim=-1)    # (K,C)
    cls_score, cls_label = torch.max(cls_prob, dim=-1)   # (K,), (K,)

    score = obj * cls_score
    keep = score >= score_thresh

    boxes = pred_boxes[keep].detach().cpu().numpy()
    scores = score[keep].detach().cpu().numpy()
    labels = cls_label[keep].detach().cpu().numpy().astype(np.int64)

    return {"boxes": boxes, "scores": scores, "labels": labels}


def draw_detections(
    img_bgr: np.ndarray,
    det: Dict[str, np.ndarray],
    class_names: List[str],
) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]

    boxes = det["boxes"]
    scores = det["scores"]
    labels = det["labels"]

    for box, s, lab in zip(boxes, scores, labels):
        x1, y1, x2, y2 = xyxy_norm_to_px(box, w, h)
        name = class_names[int(lab)] if 0 <= int(lab) < len(class_names) else str(int(lab))
        txt = f"{name} {s:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, txt, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    return out


# -------------------------
# Main inference routines
# -------------------------

@torch.no_grad()
def run_on_image(
    model: Part3Detector,
    image_path: Path,
    device: torch.device,
    image_size: int,
    score_thresh: float,
    nms_iou_thresh: float,
    class_names: List[str],
) -> np.ndarray:
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    # model expects normalized resized inputs, but we want to draw on original image
    inp = preprocess_image_bgr(img_bgr, image_size).to(device)
    out = model(inp)

    # decode K slots for the single image
    det = decode_slots(
        pred_boxes=out.pred_boxes[0],
        pred_obj_logits=out.pred_obj_logits[0],
        pred_cls_logits=out.pred_cls_logits[0],
        score_thresh=score_thresh,
    )

    # NMS (only across kept detections)
    if det["boxes"].shape[0] > 0:
        keep = nms(det["boxes"], det["scores"], nms_iou_thresh)
        det = {k: v[keep] for k, v in det.items()}

    vis = draw_detections(img_bgr, det, class_names)
    return vis


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    paths.sort()
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    ap.add_argument("--image", type=str, default=None, help="Single image path")
    ap.add_argument("--folder", type=str, default=None, help="Folder of images")
    ap.add_argument("--max-images", type=int, default=50, help="Max images to process from folder")
    ap.add_argument("--device", type=str, default=None, help="cpu or cuda (default: auto)")
    ap.add_argument("--score-thresh", type=float, default=None, help="Override score threshold")
    ap.add_argument("--nms-iou", type=float, default=None, help="Override NMS IoU threshold")
    ap.add_argument("--image-size", type=int, default=None, help="Override input image size")
    args = ap.parse_args()

    if (args.image is None) == (args.folder is None):
        raise SystemExit("Provide exactly one of --image or --folder")

    device = config.DEVICE if args.device is None else torch.device(args.device)
    image_size = int(args.image_size) if args.image_size is not None else int(config.IMAGE_SIZE)

    score_thresh = float(args.score_thresh) if args.score_thresh is not None else float(config.PART3_CONFIG.get("score_thresh", 0.25))
    nms_iou_thresh = float(args.nms_iou) if args.nms_iou is not None else float(config.PART3_CONFIG.get("nms_iou_thresh", 0.50))

    class_names = config.PART3_CONFIG.get("class_names", ["person", "car", "dog", "cat"])

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = Part3Detector().to(device)
    model.eval()
    load_checkpoint(model, ckpt_path, device)

    out_dir = Path(config.OUTPUTS_DIR) / "part3" / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image is not None:
        img_path = Path(args.image)
        vis = run_on_image(
            model=model,
            image_path=img_path,
            device=device,
            image_size=image_size,
            score_thresh=score_thresh,
            nms_iou_thresh=nms_iou_thresh,
            class_names=class_names,
        )
        out_path = out_dir / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"[OK] Saved: {out_path}")
        return

    folder = Path(args.folder)
    imgs = list_images(folder)
    if len(imgs) == 0:
        raise RuntimeError(f"No images found in: {folder}")

    n = min(args.max_images, len(imgs))
    print(f"[RUN] folder={folder}  found={len(imgs)}  processing={n}  device={device}  score={score_thresh}  nms={nms_iou_thresh}")

    for i, p in enumerate(imgs[:n]):
        vis = run_on_image(
            model=model,
            image_path=p,
            device=device,
            image_size=image_size,
            score_thresh=score_thresh,
            nms_iou_thresh=nms_iou_thresh,
            class_names=class_names,
        )
        out_path = out_dir / f"{i:04d}_{p.stem}_pred.jpg"
        cv2.imwrite(str(out_path), vis)
        if (i + 1) % 10 == 0:
            print(f"  processed {i+1}/{n}")

    print(f"[DONE] Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
