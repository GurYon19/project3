#!/usr/bin/env python3
"""
tools/build_voc_part3_k3_relaxed.py

Build a Part-3-friendly dataset index (JSON) from Pascal VOC, but **relaxed**:
- We do NOT require multi-class per image.
- We do NOT drop images with more than K objects.
  Instead, we SELECT exactly K objects per image (default K=3).

This matches the Part 3 fixed-capacity setting:
  - max_objects = K (e.g., 3)
  - minimum number of classes overall >= 2 (enforced by --classes length)

Output (by default into datasets/part3_voc_k3_relaxed/):
  train.json
  val.json
  test.json
  classes.json
  _stats.json

Sample format:
  {
    "image_id": "2007_000027",
    "image_path": "/abs/path/VOCdevkit/VOC2007/JPEGImages/2007_000027.jpg",
    "width": 486,
    "height": 500,
    "boxes": [[x1,y1,x2,y2], ...],          # len == K
    "labels": [0, 1, 2],                     # len == K
    "mask":   [true, true, false]            # len == K  (false = padded slot)
  }

Notes:
- VOC XML boxes are 1-indexed and inclusive; we convert to 0-indexed pixel coords.
- If image has < K objects after filtering, we PAD with dummy boxes [0,0,1,1],
  label = background_id, mask=False.
- If image has > K objects after filtering, we SELECT K using the chosen strategy.

Selection strategies:
- prefer_then_area (default): pick target classes first, then fill, all by area desc.
- area: pick largest area boxes, regardless of class.
- random: random K (reproducible with --seed)

"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


# ----------------------------
# Parsing helpers
# ----------------------------

@dataclass
class ObjAnn:
    name: str
    bbox: Tuple[float, float, float, float]  # x1,y1,x2,y2
    difficult: bool


def _safe_int(text: Optional[str], default: int = 0) -> int:
    try:
        return int(text) if text is not None else default
    except Exception:
        return default


def _area_xyxy(b: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def parse_voc_xml(xml_path: Path) -> Tuple[int, int, List[ObjAnn]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")

    width = _safe_int(size.findtext("width"), 0)
    height = _safe_int(size.findtext("height"), 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path} (width={width}, height={height})")

    objects: List[ObjAnn] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        difficult = _safe_int(obj.findtext("difficult"), 0) == 1
        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        xmin = _safe_int(bnd.findtext("xmin"), 0)
        ymin = _safe_int(bnd.findtext("ymin"), 0)
        xmax = _safe_int(bnd.findtext("xmax"), 0)
        ymax = _safe_int(bnd.findtext("ymax"), 0)

        # VOC is 1-indexed inclusive -> convert to 0-indexed
        x1 = max(0.0, float(xmin - 1))
        y1 = max(0.0, float(ymin - 1))
        x2 = min(float(width - 1), float(xmax - 1))
        y2 = min(float(height - 1), float(ymax - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        objects.append(ObjAnn(name=name, bbox=(x1, y1, x2, y2), difficult=difficult))

    return width, height, objects


def find_voc_root(voc_root: Path) -> Path:
    voc_root = voc_root.expanduser().resolve()

    def looks_like_voc(p: Path) -> bool:
        return (p / "JPEGImages").is_dir() and (p / "Annotations").is_dir()

    if looks_like_voc(voc_root):
        return voc_root

    for child in voc_root.iterdir():
        if child.is_dir() and looks_like_voc(child):
            return child.resolve()

    raise FileNotFoundError(f"Could not find VOC root containing JPEGImages/ and Annotations/ under: {voc_root}")


# ----------------------------
# Selection + padding
# ----------------------------

def select_k_objects(
    objs: List[ObjAnn],
    k: int,
    strategy: str,
    prefer_classes: List[str],
    rng: random.Random,
) -> List[ObjAnn]:
    if len(objs) <= k:
        return objs

    if strategy == "area":
        return sorted(objs, key=lambda o: _area_xyxy(o.bbox), reverse=True)[:k]

    if strategy == "random":
        idxs = list(range(len(objs)))
        rng.shuffle(idxs)
        return [objs[i] for i in idxs[:k]]

    if strategy == "prefer_then_area":
        preferred = [o for o in objs if o.name in set(prefer_classes)]
        others = [o for o in objs if o.name not in set(prefer_classes)]

        preferred.sort(key=lambda o: _area_xyxy(o.bbox), reverse=True)
        others.sort(key=lambda o: _area_xyxy(o.bbox), reverse=True)

        chosen = preferred[:k]
        if len(chosen) < k:
            chosen += others[: (k - len(chosen))]
        return chosen[:k]

    raise ValueError(f"Unknown selection strategy: {strategy}")


def pad_to_k(
    boxes: List[List[float]],
    labels: List[int],
    k: int,
    bg_id: int,
) -> Tuple[List[List[float]], List[int], List[bool]]:
    mask = [True] * len(boxes)
    while len(boxes) < k:
        boxes.append([0.0, 0.0, 1.0, 1.0])
        labels.append(bg_id)
        mask.append(False)
    return boxes[:k], labels[:k], mask[:k]


# ----------------------------
# Dataset build
# ----------------------------

def build_index(
    voc_root: Path,
    classes: List[str],
    max_objects: int,
    include_difficult: bool,
    allow_missing_image: bool,
    selection_strategy: str,
    seed: int,
) -> Tuple[List[Dict], Dict]:
    ann_dir = voc_root / "Annotations"
    img_dir = voc_root / "JPEGImages"

    xml_paths = sorted(ann_dir.glob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No XML files found in {ann_dir}")

    # Add background class at the end
    classes_with_bg = list(classes) + ["__background__"]
    bg_id = len(classes_with_bg) - 1
    class_to_id = {c: i for i, c in enumerate(classes_with_bg)}

    rng = random.Random(seed)

    samples: List[Dict] = []
    dropped = {
        "parse_error": 0,
        "missing_image": 0,
        "no_objects_after_filter": 0,
    }

    trunc_count = 0
    padded_count = 0

    per_class_object_counts = {c: 0 for c in classes_with_bg}
    per_class_image_counts = {c: 0 for c in classes_with_bg}
    objects_per_image_hist: Dict[int, int] = {}
    trunc_objects_per_image_hist: Dict[int, int] = {}

    for xml_path in xml_paths:
        image_id = xml_path.stem

        # Locate image file
        image_path: Optional[Path] = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = img_dir / f"{image_id}{ext}"
            if cand.exists():
                image_path = cand
                break

        if image_path is None:
            dropped["missing_image"] += 1
            if allow_missing_image:
                continue
            else:
                continue

        try:
            width, height, objs = parse_voc_xml(xml_path)
        except Exception:
            dropped["parse_error"] += 1
            continue

        # Filter to target classes (and difficulty)
        filtered: List[ObjAnn] = []
        for o in objs:
            if (not include_difficult) and o.difficult:
                continue
            if o.name not in set(classes):
                continue
            filtered.append(o)

        if len(filtered) == 0:
            dropped["no_objects_after_filter"] += 1
            continue

        # Track original count (post-filter, pre-select)
        n_before = len(filtered)
        objects_per_image_hist[n_before] = objects_per_image_hist.get(n_before, 0) + 1

        # Select up to K
        selected = select_k_objects(
            objs=filtered,
            k=max_objects,
            strategy=selection_strategy,
            prefer_classes=classes,   # prefer the target classes
            rng=rng,
        )

        if len(selected) < max_objects:
            padded_count += 1
        if n_before > max_objects:
            trunc_count += 1
            trunc_objects_per_image_hist[n_before] = trunc_objects_per_image_hist.get(n_before, 0) + 1

        boxes = [list(o.bbox) for o in selected]
        labels = [class_to_id[o.name] for o in selected]
        boxes, labels, mask = pad_to_k(boxes, labels, max_objects, bg_id)

        # Stats per class (count objects, and count images containing class)
        present_classes = set()
        for o in selected:
            per_class_object_counts[o.name] += 1
            present_classes.add(o.name)
        for c in present_classes:
            per_class_image_counts[c] += 1

        samples.append(
            {
                "image_id": image_id,
                "image_path": str(image_path.resolve()),
                "width": width,
                "height": height,
                "boxes": boxes,
                "labels": labels,
                "mask": mask,
            }
        )

    stats = {
        "voc_root": str(voc_root),
        "classes_raw": classes,
        "classes_with_bg": classes_with_bg,
        "bg_id": bg_id,
        "max_objects": max_objects,
        "include_difficult": include_difficult,
        "selection_strategy": selection_strategy,
        "seed": seed,
        "num_total_xml": len(xml_paths),
        "num_kept_samples": len(samples),
        "dropped": dropped,
        "num_truncated_images_n_gt_gt_k": trunc_count,
        "num_padded_images_n_gt_lt_k": padded_count,
        "objects_per_image_hist_post_filter": dict(sorted(objects_per_image_hist.items(), key=lambda kv: kv[0])),
        "trunc_objects_per_image_hist_post_filter": dict(sorted(trunc_objects_per_image_hist.items(), key=lambda kv: kv[0])),
        "per_class_object_counts_selected": per_class_object_counts,
        "per_class_image_counts_selected": per_class_image_counts,
    }
    return samples, stats


def split_samples(
    samples: List[Dict],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    if len(samples) < 50:
        raise ValueError(f"Too few samples after filtering ({len(samples)}).")

    rng = random.Random(seed)
    idxs = list(range(len(samples)))
    rng.shuffle(idxs)

    n = len(samples)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val

    train = [samples[i] for i in idxs[:n_train]]
    val = [samples[i] for i in idxs[n_train:n_train + n_val]]
    test = [samples[i] for i in idxs[n_train + n_val:]]

    return train, val, test


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build VOC Part-3 dataset (relaxed, select K objects).")
    p.add_argument("--voc-root", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="datasets/part3_voc_k3_relaxed")

    p.add_argument(
        "--classes",
        nargs="+",
        default=["person", "car", "dog"],
        help="Target classes (>=2 required). Background is added automatically.",
    )
    p.add_argument("--max-objects", type=int, default=3)

    p.add_argument("--include-difficult", action="store_true")
    p.add_argument("--allow-missing-image", action="store_true")

    p.add_argument(
        "--selection-strategy",
        type=str,
        default="prefer_then_area",
        choices=["prefer_then_area", "area", "random"],
        help="How to choose K GT objects when an image has >K objects after filtering.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    voc_root = find_voc_root(Path(args.voc_root))
    out_dir = Path(args.out_dir).expanduser().resolve()

    classes = [c.strip() for c in args.classes if c.strip()]
    if len(classes) < 2:
        print("[ERROR] Need at least 2 target classes for Part 3.", file=sys.stderr)
        return 2
    if args.max_objects < 1:
        print("[ERROR] max_objects must be >= 1.", file=sys.stderr)
        return 2

    print(f"[VOC] root: {voc_root}")
    print(f"[OUT] dir:  {out_dir}")
    print(f"[CFG] classes={classes}  K={args.max_objects}  include_difficult={args.include_difficult}")
    print(f"[SEL] strategy={args.selection_strategy}")
    print(f"[SPLIT] train/val/test = {args.train_ratio:.2f}/{args.val_ratio:.2f}/{args.test_ratio:.2f}  seed={args.seed}")

    samples, stats = build_index(
        voc_root=voc_root,
        classes=classes,
        max_objects=args.max_objects,
        include_difficult=args.include_difficult,
        allow_missing_image=args.allow_missing_image,
        selection_strategy=args.selection_strategy,
        seed=args.seed,
    )

    print(f"[BUILD] kept={stats['num_kept_samples']} / total_xml={stats['num_total_xml']}")
    print(f"[DROP] {stats['dropped']}")
    print(f"[INFO] truncated_images={stats['num_truncated_images_n_gt_gt_k']}  padded_images={stats['num_padded_images_n_gt_lt_k']}")
    print(f"[HIST] post_filter_objects_per_image={stats['objects_per_image_hist_post_filter']}")
    print(f"[HIST] truncated_from_counts={stats['trunc_objects_per_image_hist_post_filter']}")
    print(f"[COUNTS] per_class_object_counts_selected={stats['per_class_object_counts_selected']}")

    train, val, test = split_samples(
        samples=samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    write_json(out_dir / "train.json", train)
    write_json(out_dir / "val.json", val)
    write_json(out_dir / "test.json", test)
    write_json(out_dir / "classes.json", {"classes": stats["classes_with_bg"], "bg_id": stats["bg_id"]})
    write_json(out_dir / "_stats.json", stats)

    print(f"[OK] wrote train.json ({len(train)})")
    print(f"[OK] wrote val.json   ({len(val)})")
    print(f"[OK] wrote test.json  ({len(test)})")
    print(f"[OK] wrote classes.json (includes __background__)")
    print(f"[OK] wrote _stats.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())