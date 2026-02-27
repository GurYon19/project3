from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from part3.dataset import Part3VOCDataset, collate_part3
from part3.model import FixedSlotDetector, ModelConfig


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_classes(classes_json: str | Path) -> Tuple[List[str], int]:
    obj = json.loads(Path(classes_json).read_text(encoding="utf-8"))
    classes = obj["classes"]
    bg_id = int(obj.get("bg_id", len(classes) - 1))
    return classes, bg_id


def iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> float:
    # a,b: [4]
    ax1, ay1, ax2, ay2 = a.tolist()
    bx1, by1, bx2, by2 = b.tolist()
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-7
    return float(inter / union)


def compute_ap(rec: List[float], prec: List[float]) -> float:
    # VOC-style AP (11-point interpolation is old; here use area under PR curve)
    if not rec:
        return 0.0
    # sort by recall
    pairs = sorted(zip(rec, prec), key=lambda x: x[0])
    rec_s = [p[0] for p in pairs]
    prec_s = [p[1] for p in pairs]

    # precision envelope
    for i in range(len(prec_s) - 2, -1, -1):
        prec_s[i] = max(prec_s[i], prec_s[i + 1])

    # integrate
    ap = 0.0
    prev_r = 0.0
    for r, p in zip(rec_s, prec_s):
        ap += (r - prev_r) * p
        prev_r = r
    return float(max(0.0, ap))


@torch.no_grad()
def evaluate_map50(
    model: torch.nn.Module,
    loader: DataLoader,
    classes: List[str],
    bg_id: int,
    conf_thresh: float = 0.25,
    topk: int = 3,
    iou_thresh: float = 0.5,
    device: torch.device | None = None,
) -> Dict[str, float]:
    device = device or next(model.parameters()).device
    model.eval()

    # Collect GT per image per class
    gt: Dict[str, Dict[int, List[torch.Tensor]]] = {}
    # Collect predictions per class: list of (image_id, score, box)
    preds: Dict[int, List[Tuple[str, float, torch.Tensor]]] = {c: [] for c in range(len(classes)) if c != bg_id}

    for images, targets in loader:
        images = images.to(device)
        out = model(images)
        boxes = out["boxes"].cpu()        # [B,K,4]
        logits = out["logits"].cpu()      # [B,K,C]
        probs = torch.softmax(logits, dim=-1)

        B, K, C = probs.shape

        for b in range(B):
            image_id = targets["meta"][b]["image_id"]
            tb = targets["boxes"][b]      # [K,4]
            tl = targets["labels"][b]     # [K]
            tm = targets["mask"][b]       # [K]

            gt.setdefault(image_id, {})
            # add GT objects
            for j in range(K):
                if not bool(tm[j]):
                    continue
                cls = int(tl[j].item())
                if cls == bg_id:
                    continue
                gt[image_id].setdefault(cls, [])
                gt[image_id][cls].append(tb[j].cpu())

            # predictions: topk by best non-bg class confidence
            for k in range(K):
                p = probs[b, k]
                # best class excluding bg
                p_bg = float(p[bg_id].item())
                p2 = p.clone()
                p2[bg_id] = -1.0
                cls = int(torch.argmax(p2).item())
                score = float(p[cls].item())
                if score < conf_thresh:
                    continue
                preds[cls].append((image_id, score, boxes[b, k]))

    # Compute AP per class
    ap_per_class: Dict[str, float] = {}
    for cls in sorted(preds.keys()):
        pred_list = sorted(preds[cls], key=lambda x: x[1], reverse=True)
        n_gt = sum(len(gt.get(img, {}).get(cls, [])) for img in gt.keys())
        if n_gt == 0:
            ap_per_class[classes[cls]] = 0.0
            continue

        used: Dict[str, List[bool]] = {}
        for img_id in gt:
            used[img_id] = [False] * len(gt.get(img_id, {}).get(cls, []))

        tp = []
        fp = []

        for img_id, score, box_pred in pred_list:
            gts = gt.get(img_id, {}).get(cls, [])
            if not gts:
                fp.append(1.0)
                tp.append(0.0)
                continue

            # match best IoU unused GT
            best_iou = 0.0
            best_j = -1
            for j, box_gt in enumerate(gts):
                if used[img_id][j]:
                    continue
                i = iou_xyxy(box_pred, box_gt)
                if i > best_iou:
                    best_iou = i
                    best_j = j

            if best_iou >= iou_thresh and best_j >= 0:
                used[img_id][best_j] = True
                tp.append(1.0)
                fp.append(0.0)
            else:
                tp.append(0.0)
                fp.append(1.0)

        # PR curve
        tp_cum = []
        fp_cum = []
        s_tp = 0.0
        s_fp = 0.0
        for t, f in zip(tp, fp):
            s_tp += t
            s_fp += f
            tp_cum.append(s_tp)
            fp_cum.append(s_fp)

        rec = [t / max(1.0, n_gt) for t in tp_cum]
        prec = [t / max(1.0, (t + f)) for t, f in zip(tp_cum, fp_cum)]
        ap = compute_ap(rec, prec)
        ap_per_class[classes[cls]] = ap

    # mAP over non-bg classes present
    valid_classes = [c for c in ap_per_class.keys() if c != "__background__"]
    mAP = sum(ap_per_class[c] for c in valid_classes) / max(1, len(valid_classes))
    ap_per_class["mAP@0.5"] = mAP
    return ap_per_class


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--max-objects", type=int, default=3)
    p.add_argument("--conf-thresh", type=float, default=0.25)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/part3")
    p.add_argument("--tag", type=str, default="eval")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device()
    print(f"[DEVICE] {device}")

    data_dir = Path(args.data-dir) if isinstance(args.data_dir, str) else Path(args.data_dir)
    split_json = Path(args.data_dir) / f"{args.split}.json"
    classes_json = Path(args.data_dir) / "classes.json"

    classes, bg_id = load_classes(classes_json)
    print(f"[DATA] split={args.split} classes={classes} bg_id={bg_id}")

    ds = Part3VOCDataset(split_json, classes_json, image_size=args.image_size, max_objects=args.max_objects, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_part3)

    cfg = ModelConfig(
        image_size=args.image_size,
        max_objects=args.max_objects,
        num_classes_total=len(classes),
        bg_id=bg_id,
        backbone="mobilenet_v3_small",
        pretrained=False,
    )
    model = FixedSlotDetector(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    ap = evaluate_map50(model, loader, classes, bg_id, conf_thresh=args.conf_thresh, topk=args.topk, device=device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"metrics_{args.tag}.json"
    out_path.write_text(json.dumps(ap, indent=2), encoding="utf-8")

    print("\n=== AP@0.5 Results ===")
    for k, v in ap.items():
        print(f"{k:>12s}  {v:.4f}")
    print(f"\n[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()