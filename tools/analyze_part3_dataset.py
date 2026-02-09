#!/usr/bin/env python3
"""
Analyze the filtered Part 3 COCO dataset in data/part3.

Computes:
- images per split
- boxes per class
- objects-per-image histogram
- multi-class-in-image counts
- class co-occurrence counts
- warnings for rare classes (help decide dropping 'traffic light')
Writes data/part3/_stats.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from itertools import combinations


def load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def analyze_split(split_root: Path) -> dict:
    ann_path = split_root / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing {ann_path}")

    coco = load_json(ann_path)
    categories = coco.get("categories", [])
    id_to_name = {c["id"]: c["name"] for c in categories}

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    anns_by_img = defaultdict(list)
    for ann in annotations:
        anns_by_img[ann["image_id"]].append(ann)

    obj_per_img = Counter()
    class_box_counts = Counter()
    class_img_presence = Counter()  # how many images contain each class at least once
    multi_class_images = 0
    cooccur = Counter()

    for im in images:
        img_id = im["id"]
        anns = anns_by_img.get(img_id, [])
        obj_per_img[len(anns)] += 1

        present_classes = set()
        for a in anns:
            cid = a["category_id"]
            class_box_counts[cid] += 1
            present_classes.add(cid)

        for cid in present_classes:
            class_img_presence[cid] += 1

        if len(present_classes) >= 2:
            multi_class_images += 1
            for a, b in combinations(sorted(present_classes), 2):
                cooccur[(a, b)] += 1

    # make readable outputs
    class_box_counts_named = {id_to_name[k]: v for k, v in class_box_counts.items()}
    class_img_presence_named = {id_to_name[k]: v for k, v in class_img_presence.items()}
    cooccur_named = {
        f"{id_to_name[a]} + {id_to_name[b]}": v for (a, b), v in cooccur.items()
    }

    return {
        "split": split_root.name,
        "num_images": len(images),
        "num_annotations": len(annotations),
        "objects_per_image_hist": dict(sorted(obj_per_img.items())),
        "boxes_per_class": dict(sorted(class_box_counts_named.items(), key=lambda x: -x[1])),
        "images_with_class_present": dict(sorted(class_img_presence_named.items(), key=lambda x: -x[1])),
        "num_images_with_2plus_classes": multi_class_images,
        "cooccurrence_counts": dict(sorted(cooccur_named.items(), key=lambda x: -x[1])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/part3",
                    help="Filtered Part 3 dataset root (contains train/valid/test).")
    ap.add_argument("--out", type=str, default="data/part3/_stats.json",
                    help="Where to write stats JSON.")
    ap.add_argument("--rare-box-threshold", type=int, default=300,
                    help="Warn if a class has fewer than this many boxes in TRAIN.")
    ap.add_argument("--rare-image-threshold", type=float, default=0.05,
                    help="Warn if a class appears in fewer than this fraction of TRAIN images.")
    args = ap.parse_args()

    root = Path(args.root)
    splits = ["train", "valid", "test"]
    for s in splits:
        if not (root / s / "annotations.json").exists():
            raise FileNotFoundError(f"Missing filtered split annotations: {root / s / 'annotations.json'}")

    stats = {"root": str(root), "splits": {}}
    for s in splits:
        stats["splits"][s] = analyze_split(root / s)

    # Warnings based on TRAIN only (for dropping classes decisions)
    train = stats["splits"]["train"]
    train_images = train["num_images"]
    boxes = train["boxes_per_class"]
    present = train["images_with_class_present"]

    warnings = []
    for cls_name, box_count in boxes.items():
        img_count = present.get(cls_name, 0)
        frac = (img_count / train_images) if train_images > 0 else 0.0

        if box_count < args.rare_box_threshold:
            warnings.append(
                f"RARE (boxes): '{cls_name}' has {box_count} boxes in TRAIN (< {args.rare_box_threshold}). Consider dropping."
            )
        if frac < args.rare_image_threshold:
            warnings.append(
                f"RARE (images): '{cls_name}' appears in {img_count}/{train_images} TRAIN images ({frac:.3f} < {args.rare_image_threshold}). Consider dropping."
            )

    # Also warn if multi-class images are too few
    mc = train["num_images_with_2plus_classes"]
    if train_images > 0 and (mc / train_images) < 0.10:
        warnings.append(
            f"LOW MULTI-CLASS: only {mc}/{train_images} train images have ≥2 classes (<10%). "
            f"This can weaken multi-class learning."
        )

    stats["warnings"] = warnings
    save_json(stats, Path(args.out))

    print(f"[DONE] Wrote stats to {args.out}")
    print("\n=== TRAIN SUMMARY ===")
    print(f"Images: {train_images}, Anns: {train['num_annotations']}")
    print(f"Objects/image hist: {train['objects_per_image_hist']}")
    print(f"Images with ≥2 classes: {mc} ({mc/train_images:.2%} of train)" if train_images else "Images with ≥2 classes: 0")
    print("\nBoxes per class (train):")
    for k, v in boxes.items():
        print(f"  {k:15s} {v}")
    if warnings:
        print("\n=== WARNINGS ===")
        for w in warnings:
            print(" -", w)


if __name__ == "__main__":
    main()
