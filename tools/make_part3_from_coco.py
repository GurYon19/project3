#!/usr/bin/env python3
"""
Filter a Roboflow-exported COCO dataset to a Part-3-ready subset.

- Keeps only chosen classes (by category name)
- Drops images with 0 chosen objects
- Drops images with > max_objects chosen objects (default 3)
- Remaps category ids to contiguous [0..C-1]
- Copies kept images into data/part3/{split}/images/
- Writes annotations.json per split + classes.json

Input structure expected:
data/raw_coco/{train,valid,test}/_annotations.coco.json
data/raw_coco/{train,valid,test}/*.jpg

Output structure:
data/part3/{train,valid,test}/images/*.jpg
data/part3/{train,valid,test}/annotations.json
data/part3/classes.json
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_CLASSES = ["person", "car", "bicycle", "dog", "cat", "traffic light"]


def load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def build_name_to_cat(coco: dict) -> Dict[str, dict]:
    name_to_cat = {}
    for c in coco.get("categories", []):
        # COCO uses names like "traffic light" (space included)
        name_to_cat[c["name"]] = c
    return name_to_cat


def ensure_unique(names: List[str]) -> None:
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate class names provided: {names}")


def filter_split(
    split_dir: Path,
    out_dir: Path,
    chosen_names: List[str],
    max_objects: int,
    copy_mode: str,
) -> Tuple[dict, dict]:
    """
    Returns:
      (filtered_coco, split_stats)
    """
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")

    coco = load_json(ann_path)
    name_to_cat = build_name_to_cat(coco)

    # Validate classes exist
    missing = [n for n in chosen_names if n not in name_to_cat]
    if missing:
        available = sorted(list(name_to_cat.keys()))
        raise ValueError(
            f"These chosen classes were not found in categories: {missing}\n"
            f"Available category names include (sample): {available[:30]}"
        )

    chosen_cat_ids = {name_to_cat[n]["id"] for n in chosen_names}

    # Build image_id -> image record
    images = coco.get("images", [])
    img_by_id = {im["id"]: im for im in images}

    # Group annotations by image_id, only for chosen classes
    anns_by_img = defaultdict(list)
    kept_ann_count = 0
    for ann in coco.get("annotations", []):
        if ann.get("category_id") in chosen_cat_ids:
            img_id = ann["image_id"]
            anns_by_img[img_id].append(ann)
            kept_ann_count += 1

    # Decide which images to keep based on chosen-class object count
    kept_img_ids = []
    dropped_zero = 0
    dropped_too_many = 0
    obj_hist = defaultdict(int)

    for img_id, im in img_by_id.items():
        k = len(anns_by_img.get(img_id, []))
        if k == 0:
            dropped_zero += 1
            continue
        if k > max_objects:
            dropped_too_many += 1
            continue
        kept_img_ids.append(img_id)
        obj_hist[k] += 1

    kept_img_ids_set = set(kept_img_ids)

    # Remap categories to contiguous ids 0..C-1 in the order of chosen_names
    model_id_by_name = {name: i for i, name in enumerate(chosen_names)}
    oldcat_to_newcat = {name_to_cat[name]["id"]: model_id_by_name[name] for name in chosen_names}

    new_categories = [{"id": model_id_by_name[name], "name": name} for name in chosen_names]

    # Build new images list (same records)
    new_images = [img_by_id[i] for i in kept_img_ids]

    # Build new annotations list with remapped category_id and new annotation ids
    new_annotations = []
    new_ann_id = 1
    bad_box = 0

    for img_id in kept_img_ids:
        for ann in anns_by_img[img_id]:
            # Validate bbox sanity quickly
            bbox = ann.get("bbox", None)
            if not bbox or len(bbox) != 4:
                bad_box += 1
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                bad_box += 1
                continue

            ann2 = dict(ann)
            ann2["id"] = new_ann_id
            ann2["category_id"] = oldcat_to_newcat[ann["category_id"]]
            new_annotations.append(ann2)
            new_ann_id += 1

    # Copy / link images
    out_images_dir = out_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for im in new_images:
        src = split_dir / im["file_name"]
        dst = out_images_dir / im["file_name"]

        if not src.exists():
            # Some exports may store images in an "images" subfolder; try that
            alt = split_dir / "images" / im["file_name"]
            if alt.exists():
                src = alt
            else:
                raise FileNotFoundError(f"Missing image file: {src} (and {alt})")

        dst.parent.mkdir(parents=True, exist_ok=True)
        if copy_mode == "copy":
            shutil.copy2(src, dst)
        elif copy_mode == "hardlink":
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
        else:
            raise ValueError("copy_mode must be 'copy' or 'hardlink'")
        copied += 1

    filtered = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": new_categories,
    }

    split_stats = {
        "split": split_dir.name,
        "chosen_classes": chosen_names,
        "max_objects": max_objects,
        "raw_images": len(images),
        "raw_annotations": len(coco.get("annotations", [])),
        "chosen_class_annotations_raw": kept_ann_count,
        "kept_images": len(new_images),
        "kept_annotations": len(new_annotations),
        "dropped_images_zero_objects": dropped_zero,
        "dropped_images_over_max_objects": dropped_too_many,
        "objects_per_image_hist": dict(sorted(obj_hist.items())),
        "bad_or_skipped_bboxes": bad_box,
        "copy_mode": copy_mode,
    }
    return filtered, split_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=str, default="data/raw_coco",
                    help="Path containing train/valid/test folders from Roboflow export.")
    ap.add_argument("--out-root", type=str, default="data/part3",
                    help="Output dataset root for filtered Part 3 dataset.")
    ap.add_argument("--classes", type=str, default=",".join(DEFAULT_CLASSES),
                    help="Comma-separated class names (must match COCO category 'name').")
    ap.add_argument("--max-objects", type=int, default=3,
                    help="Max number of chosen-class objects allowed per image (capacity).")
    ap.add_argument("--copy-mode", type=str, default="copy", choices=["copy", "hardlink"],
                    help="copy = duplicate images; hardlink = fast, saves disk (same drive only).")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    chosen_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    ensure_unique(chosen_names)

    # Verify split dirs exist
    splits = ["train", "valid", "test"]
    for s in splits:
        if not (raw_root / s).exists():
            raise FileNotFoundError(f"Missing split dir: {raw_root / s}")

    out_root.mkdir(parents=True, exist_ok=True)

    all_stats = {"config": {"classes": chosen_names, "max_objects": args.max_objects}, "splits": {}}

    for s in splits:
        split_dir = raw_root / s
        out_dir = out_root / s
        filtered, stats = filter_split(
            split_dir=split_dir,
            out_dir=out_dir,
            chosen_names=chosen_names,
            max_objects=args.max_objects,
            copy_mode=args.copy_mode,
        )

        save_json(filtered, out_dir / "annotations.json")
        all_stats["splits"][s] = stats
        print(f"[OK] {s}: kept {stats['kept_images']} images, {stats['kept_annotations']} anns "
              f"(dropped 0-obj={stats['dropped_images_zero_objects']}, "
              f"dropped >{args.max_objects}={stats['dropped_images_over_max_objects']})")

    # Write classes.json
    classes_obj = {
        "num_classes": len(chosen_names),
        "id_to_name": {str(i): name for i, name in enumerate(chosen_names)},
        "name_to_id": {name: i for i, name in enumerate(chosen_names)},
        "max_objects": args.max_objects,
    }
    save_json(classes_obj, out_root / "classes.json")
    save_json(all_stats, out_root / "_filter_stats.json")
    print(f"[DONE] Wrote filtered dataset to: {out_root}")
    print(f"[DONE] Wrote filter stats: {out_root / '_filter_stats.json'}")
    print(f"[DONE] Wrote classes mapping: {out_root / 'classes.json'}")


if __name__ == "__main__":
    main()
