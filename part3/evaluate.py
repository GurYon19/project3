# part3/evaluate.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from part3.dataset import Part3VOCDataset, collate_part3
from part3.model import FixedSlotDetector


# -------------------------
# Box utils
# -------------------------

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    IoU between two boxes in xyxy (pixel coords).
    a, b: [4]
    """
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


def clamp_xyxy(box: np.ndarray, W: int, H: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1 = np.clip(x1, 0, W - 1)
    y1 = np.clip(y1, 0, H - 1)
    x2 = np.clip(x2, 0, W - 1)
    y2 = np.clip(y2, 0, H - 1)
    # ensure non-degenerate
    x2 = max(x2, x1 + 1.0)
    y2 = max(y2, y1 + 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# -------------------------
# AP computation (VOC-style, continuous)
# -------------------------

def average_precision(rec: np.ndarray, prec: np.ndarray) -> float:
    """
    Continuous AP (VOC2010+ style):
      - precision envelope
      - integrate over recall
    """
    if rec.size == 0:
        return 0.0

    # Add sentinel endpoints
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # Precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Integrate where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


@dataclass
class Pred:
    image_idx: int
    cls: int
    score: float
    box: np.ndarray  # [4]


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(model: torch.nn.Module, ckpt_path: str | Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    num_classes: int,
    bg_id: int,
    image_size: int,
    conf_thresh: float,
    topk: int,
) -> Tuple[List[Pred], Dict[int, Dict[int, List[np.ndarray]]]]:
    """
    Returns:
      preds: list of Pred across all images
      gts: dict[class_id][image_idx] -> list of GT boxes for that class in that image
    """
    model.eval()
    preds: List[Pred] = []

    # gts[class][img] = [boxes...]
    gts: Dict[int, Dict[int, List[np.ndarray]]] = {c: {} for c in range(num_classes)}

    img_counter = 0
    for images, targets in loader:
        B = images.shape[0]
        images = images.to(next(model.parameters()).device)

        out = model(images)
        boxes = out["boxes"].detach().cpu().numpy()   # [B,K,4]
        logits = out["logits"].detach().cpu()         # [B,K,C+1]
        probs = F.softmax(logits, dim=-1).numpy()     # [B,K,C+1]

        # GT collection
        tgt_boxes = targets["boxes"].numpy()          # [B,K,4]
        tgt_labels = targets["labels"].numpy()        # [B,K]
        tgt_mask = targets["mask"].numpy()            # [B,K]

        for bi in range(B):
            img_idx = img_counter + bi

            # GT: only valid
            valid = tgt_mask[bi].astype(bool)
            gt_b = tgt_boxes[bi][valid]
            gt_l = tgt_labels[bi][valid]
            for j in range(len(gt_l)):
                c = int(gt_l[j])
                if c == bg_id:
                    continue
                gts[c].setdefault(img_idx, []).append(
                    clamp_xyxy(gt_b[j], image_size, image_size)
                )

            # Predictions: take topk non-background above threshold
            # Score = max prob over non-bg classes, cls = argmax over non-bg classes
            p = probs[bi]  # [K,C+1]
            p_nobg = p[:, :num_classes]  # exclude bg column
            cls = np.argmax(p_nobg, axis=1)           # [K]
            score = np.max(p_nobg, axis=1)            # [K]

            # filter
            cand = [(k, float(score[k])) for k in range(p_nobg.shape[0]) if float(score[k]) >= conf_thresh]
            cand.sort(key=lambda x: x[1], reverse=True)
            cand = cand[:topk]

            for k, sc in cand:
                c = int(cls[k])
                box = clamp_xyxy(boxes[bi, k], image_size, image_size)
                preds.append(Pred(image_idx=img_idx, cls=c, score=sc, box=box))

        img_counter += B

    return preds, gts


def evaluate_ap50(
    preds: List[Pred],
    gts: Dict[int, Dict[int, List[np.ndarray]]],
    num_classes: int,
    iou_thresh: float = 0.5,
) -> Dict:
    """
    VOC-style AP@0.5 per class + mAP.
    """
    results = {
        "iou_thresh": iou_thresh,
        "per_class": {},
        "mAP": None,
    }

    aps = []

    for c in range(num_classes):
        # all GT boxes for this class
        gt_for_c = gts.get(c, {})
        n_gt = sum(len(v) for v in gt_for_c.values())

        # predictions for this class
        pred_c = [p for p in preds if p.cls == c]
        pred_c.sort(key=lambda p: p.score, reverse=True)

        # matched flags per image gt box
        matched = {img: np.zeros(len(boxes), dtype=bool) for img, boxes in gt_for_c.items()}

        tp = np.zeros(len(pred_c), dtype=np.float32)
        fp = np.zeros(len(pred_c), dtype=np.float32)

        for i, p in enumerate(pred_c):
            gt_boxes = gt_for_c.get(p.image_idx, [])
            if len(gt_boxes) == 0:
                fp[i] = 1.0
                continue

            # find best unmatched GT IoU
            best_iou = 0.0
            best_j = -1
            for j, gt_box in enumerate(gt_boxes):
                if matched[p.image_idx][j]:
                    continue
                v = iou_xyxy(p.box, gt_box)
                if v > best_iou:
                    best_iou = v
                    best_j = j

            if best_iou >= iou_thresh and best_j >= 0:
                tp[i] = 1.0
                matched[p.image_idx][best_j] = True
            else:
                fp[i] = 1.0

        # precision-recall
        if len(pred_c) == 0:
            ap = 0.0
            prec = np.array([], dtype=np.float32)
            rec = np.array([], dtype=np.float32)
        else:
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            rec = tp_cum / max(n_gt, 1)

            ap = average_precision(rec, prec)

        results["per_class"][str(c)] = {
            "num_gt": int(n_gt),
            "num_pred": int(len(pred_c)),
            "AP@0.5": float(ap),
            "precision_last": float(prec[-1]) if prec.size else 0.0,
            "recall_last": float(rec[-1]) if rec.size else 0.0,
        }
        aps.append(ap)

    results["mAP"] = float(np.mean(aps)) if len(aps) else 0.0
    return results


def parse_args():
    p = argparse.ArgumentParser("Part3 evaluation (VOC-style AP@0.5) for fixed-slot detector")
    p.add_argument("--data-dir", type=str, default="datasets/part3")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--max-objects", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--conf-thresh", type=float, default=0.35, help="Filter predictions by score")
    p.add_argument("--topk", type=int, default=3, help="Keep at most top-k predictions per image")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for AP (default 0.5)")

    p.add_argument("--out-dir", type=str, default="outputs/part3")
    p.add_argument("--tag", type=str, default="eval_run1", help="Name tag for output json")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device()

    data_dir = Path(args.data_dir)
    index_json = data_dir / f"{args.split}.json"
    classes_json = data_dir / "classes.json"

    ds = Part3VOCDataset(index_json, classes_json, image_size=args.image_size, max_objects=args.max_objects)
    num_classes = len(ds.classes)
    bg_id = num_classes

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_part3,
        pin_memory=(device.type == "cuda"),
    )

    model = FixedSlotDetector(
        num_classes=num_classes,
        max_objects=args.max_objects,
        pretrained=False,
        freeze_backbone=False,
        image_size=args.image_size,
    ).to(device)

    load_checkpoint(model, args.checkpoint, device)

    print(f"[DEVICE] {device}")
    print(f"[DATA] split={args.split} n={len(ds)} classes={ds.classes} bg_id={bg_id}")
    print(f"[EVAL] conf_thresh={args.conf_thresh} topk={args.topk} iou={args.iou}")

    preds, gts = collect_predictions(
        model=model,
        loader=loader,
        num_classes=num_classes,
        bg_id=bg_id,
        image_size=args.image_size,
        conf_thresh=args.conf_thresh,
        topk=args.topk,
    )

    res = evaluate_ap50(preds, gts, num_classes=num_classes, iou_thresh=args.iou)

    # Attach class names for readability
    res["class_names"] = ds.classes
    res["split"] = args.split
    res["checkpoint"] = str(Path(args.checkpoint).resolve())
    res["conf_thresh"] = float(args.conf_thresh)
    res["topk"] = int(args.topk)

    print("\n=== AP@0.5 Results ===")
    for i, name in enumerate(ds.classes):
        row = res["per_class"][str(i)]
        print(f"{name:>10s}  AP@0.5={row['AP@0.5']:.4f}  (GT={row['num_gt']}, Pred={row['num_pred']})")
    print(f"\nmAP@0.5 = {res['mAP']:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}_{args.split}_ap50.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
