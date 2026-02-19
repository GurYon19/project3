#!/usr/bin/env python3
"""
tools/filter_voc_for_part3.py

Builds a Part-3-friendly dataset index (JSON) from a Pascal VOC-style folder.

Goal:
- Multi-class, multi-object detection with fixed capacity (default max_objects=3).
- Keep original VOC files untouched (Option B). We only write JSON index files.

Expected VOC directory layout (minimum):
  VOC_ROOT/
    JPEGImages/
    Annotations/
    ImageSets/Main/   (optional; not required)

Outputs (by default into datasets/part3/):
  train.json
  val.json
  test.json
  classes.json
  _stats.json

Each JSON is a list of samples:
  {
    "image_id": "2007_000027",
    "image_path": "/abs/or/relative/path/to/JPEGImages/2007_000027.jpg",
    "width": 486,
    "height": 500,
    "boxes": [[x1,y1,x2,y2], ...],          # float (pixel coords)
    "labels": [0, 1, 2],                     # int indices into classes.json
  }

Notes:
- VOC boxes are 1-indexed and inclusive in XML (xmin/ymin/xmax/ymax). We convert to 0-indexed, and
  keep them as [x1,y1,x2,y2] in pixel coordinates.
- By default we drop objects marked <difficult>==1 (can include with --include-difficult).
- We filter to selected classes and enforce 1 <= num_objects <= max_objects after filtering.
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
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    difficult: bool


def _safe_int(text: Optional[str], default: int = 0) -> int:
    try:
        return int(text) if text is not None else default
    except Exception:
        return default


def parse_voc_xml(xml_path: Path) -> Tuple[int, int, List[ObjAnn]]:
    """
    Returns: (width, height, objects)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")

    width = _safe_int(size.findtext("width"), 0)
    height = _safe_int(size.findtext("height"), 0)
    if width <= 0 or height <= 0:
        # Some VOC variants may omit size; in that case you can extend this script
        # to read from the image. Here we keep it strict.
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

        # VOC is 1-indexed inclusive; convert to 0-indexed.
        # Also clamp to image bounds.
        x1 = max(0.0, float(xmin - 1))
        y1 = max(0.0, float(ymin - 1))
        x2 = min(float(width - 1), float(xmax - 1))
        y2 = min(float(height - 1), float(ymax - 1))

        # Validate box
        if x2 <= x1 or y2 <= y1:
            continue

        objects.append(ObjAnn(name=name, bbox=(x1, y1, x2, y2), difficult=difficult))

    return width, height, objects


def find_voc_root(voc_root: Path) -> Path:
    """
    Attempts to locate the folder that actually contains JPEGImages/Annotations.
    This helps when the user has an extra nesting level after manual download.
    """
    voc_root = voc_root.expanduser().resolve()

    def looks_like_voc(p: Path) -> bool:
        return (p / "JPEGImages").is_dir() and (p / "Annotations").is_dir()

    if looks_like_voc(voc_root):
        return voc_root

    # Common nesting: VOC2012_train_val/VOC2012_train_val/...
    for child in voc_root.iterdir():
        if child.is_dir() and looks_like_voc(child):
            return child.resolve()

    raise FileNotFoundError(
        f"Could not find VOC root containing JPEGImages/ and Annotations/ under: {voc_root}"
    )


# ----------------------------
# Dataset building
# ----------------------------

def build_index(
    voc_root: Path,
    classes: List[str],
    max_objects: int,
    include_difficult: bool,
    allow_missing_jpg: bool,
) -> Tuple[List[Dict], Dict]:
    """
    Walk all annotation XMLs, filter to classes, enforce 1..max_objects objects, and return samples.
    Also returns stats dict.
    """
    ann_dir = voc_root / "Annotations"
    img_dir = voc_root / "JPEGImages"

    class_to_id = {c: i for i, c in enumerate(classes)}

    xml_paths = sorted(ann_dir.glob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No XML files found in {ann_dir}")

    samples: List[Dict] = []
    dropped = {
        "no_objects_after_filter": 0,
        "too_many_objects": 0,
        "missing_image": 0,
        "parse_error": 0,
    }

    per_class_counts = {c: 0 for c in classes}
    objects_per_image_hist: Dict[int, int] = {}

    for xml_path in xml_paths:
        image_id = xml_path.stem
        jpg_path = img_dir / f"{image_id}.jpg"
        jpeg_path = img_dir / f"{image_id}.jpeg"
        png_path = img_dir / f"{image_id}.png"

        image_path: Optional[Path] = None
        for cand in (jpg_path, jpeg_path, png_path):
            if cand.exists():
                image_path = cand
                break

        if image_path is None:
            dropped["missing_image"] += 1
            if allow_missing_jpg:
                continue
            else:
                # strict by default: missing image indicates broken dataset
                continue

        try:
            width, height, objs = parse_voc_xml(xml_path)
        except Exception:
            dropped["parse_error"] += 1
            continue

        # Filter objects
        filtered: List[ObjAnn] = []
        for o in objs:
            if (not include_difficult) and o.difficult:
                continue
            if o.name not in class_to_id:
                continue
            filtered.append(o)

        n = len(filtered)
        if n == 0:
            dropped["no_objects_after_filter"] += 1
            continue
        if n > max_objects:
            dropped["too_many_objects"] += 1
            continue

        boxes = [list(o.bbox) for o in filtered]
        labels = [class_to_id[o.name] for o in filtered]

        # Stats
        for o in filtered:
            per_class_counts[o.name] += 1
        objects_per_image_hist[n] = objects_per_image_hist.get(n, 0) + 1

        samples.append(
            {
                "image_id": image_id,
                "image_path": str(image_path.resolve()),
                "width": width,
                "height": height,
                "boxes": boxes,
                "labels": labels,
            }
        )

    stats = {
        "voc_root": str(voc_root),
        "classes": classes,
        "max_objects": max_objects,
        "include_difficult": include_difficult,
        "num_total_xml": len(xml_paths),
        "num_kept_samples": len(samples),
        "dropped": dropped,
        "per_class_object_counts": per_class_counts,
        "objects_per_image_hist": dict(sorted(objects_per_image_hist.items(), key=lambda kv: kv[0])),
    }
    return samples, stats


def split_samples(
    samples: List[Dict],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if len(samples) < 10:
        raise ValueError(f"Too few samples after filtering ({len(samples)}). Pick more classes or raise max_objects.")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    rng = random.Random(seed)
    idxs = list(range(len(samples)))
    rng.shuffle(idxs)

    n = len(samples)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # ensure all accounted for
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val

    train = [samples[i] for i in idxs[:n_train]]
    val = [samples[i] for i in idxs[n_train:n_train + n_val]]
    test = [samples[i] for i in idxs[n_train + n_val:]]

    assert len(train) + len(val) + len(test) == n
    return train, val, test


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter Pascal VOC and build Part-3 JSON indices.")
    p.add_argument(
        "--voc-root",
        type=str,
        required=True,
        help="Path to the folder containing JPEGImages/ and Annotations/ (or a parent folder with one extra nesting).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="datasets/part3",
        help="Output directory for train/val/test JSON files (default: datasets/part3).",
    )
    p.add_argument(
        "--classes",
        nargs="+",
        default=["person", "car", "dog"],
        help="List of classes to keep (default: person car dog).",
    )
    p.add_argument(
        "--max-objects",
        type=int,
        default=3,
        help="Maximum objects per image AFTER filtering (default: 3).",
    )
    p.add_argument(
        "--include-difficult",
        action="store_true",
        help="Include objects with <difficult> == 1 (default: excluded).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42).",
    )
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument(
        "--allow-missing-image",
        action="store_true",
        help="If an annotation XML exists but the image file is missing, skip it instead of treating it as fatal.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    voc_root = find_voc_root(Path(args.voc_root))
    out_dir = Path(args.out_dir).expanduser().resolve()

    classes = [c.strip() for c in args.classes if c.strip()]
    if len(classes) < 2:
        print("[ERROR] Need at least 2 classes for Part 3.", file=sys.stderr)
        return 2
    if args.max_objects < 1:
        print("[ERROR] max_objects must be >= 1.", file=sys.stderr)
        return 2

    print(f"[VOC] root: {voc_root}")
    print(f"[OUT] dir:  {out_dir}")
    print(f"[CFG] classes={classes}  max_objects={args.max_objects}  include_difficult={args.include_difficult}")
    print(f"[SPLIT] train/val/test = {args.train_ratio:.2f}/{args.val_ratio:.2f}/{args.test_ratio:.2f}  seed={args.seed}")

    samples, stats = build_index(
        voc_root=voc_root,
        classes=classes,
        max_objects=args.max_objects,
        include_difficult=args.include_difficult,
        allow_missing_jpg=args.allow_missing_image,
    )

    print(f"[FILTER] kept_samples={stats['num_kept_samples']}  total_xml={stats['num_total_xml']}")
    print(f"[FILTER] dropped={stats['dropped']}")
    print(f"[STATS] objects_per_image_hist={stats['objects_per_image_hist']}")
    print(f"[STATS] per_class_object_counts={stats['per_class_object_counts']}")

    train, val, test = split_samples(
        samples=samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    # Write outputs
    write_json(out_dir / "train.json", train)
    write_json(out_dir / "val.json", val)
    write_json(out_dir / "test.json", test)
    write_json(out_dir / "classes.json", {"classes": classes})
    write_json(out_dir / "_stats.json", stats)

    print(f"[OK] wrote: {out_dir / 'train.json'}  ({len(train)} samples)")
    print(f"[OK] wrote: {out_dir / 'val.json'}    ({len(val)} samples)")
    print(f"[OK] wrote: {out_dir / 'test.json'}   ({len(test)} samples)")
    print(f"[OK] wrote: {out_dir / 'classes.json'}")
    print(f"[OK] wrote: {out_dir / '_stats.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
