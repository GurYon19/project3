#!/usr/bin/env python3
"""
tools/voc_part3_class_stats.py

Compute per-class stats for *all* classes in a Pascal VOC-style dataset,
*after applying Part-3-style filtering*:

- Optionally exclude <difficult> objects
- Apply max_objects AFTER filtering (i.e., drop images that would exceed capacity)
- Output:
  - total discovered classes (raw)
  - per-class object counts (after filtering)
  - per-class image counts (after filtering)
  - objects-per-image histogram (after filtering)
  - pairwise co-occurrence counts (after filtering) (optional)

This script does NOT modify any dataset files and does NOT depend on your existing filter script.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
    difficult: bool


def _safe_int(text: Optional[str], default: int = 0) -> int:
    try:
        return int(text) if text is not None else default
    except Exception:
        return default


def parse_voc_xml_names(xml_path: Path) -> List[ObjAnn]:
    """Return list of (class_name, difficult) for objects in VOC XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objs: List[ObjAnn] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if not name:
            continue
        difficult = _safe_int(obj.findtext("difficult"), 0) == 1
        objs.append(ObjAnn(name=name, difficult=difficult))
    return objs


def find_voc_root(voc_root: Path) -> Path:
    """Locate folder containing JPEGImages/ and Annotations/ (supports one nesting level)."""
    voc_root = voc_root.expanduser().resolve()

    def looks_like_voc(p: Path) -> bool:
        return (p / "Annotations").is_dir()

    if looks_like_voc(voc_root):
        return voc_root

    for child in voc_root.iterdir():
        if child.is_dir() and looks_like_voc(child):
            return child.resolve()

    raise FileNotFoundError(
        f"Could not find VOC root containing Annotations/ under: {voc_root}"
    )


# ----------------------------
# Stats computation
# ----------------------------

def compute_stats(
    voc_root: Path,
    max_objects: int,
    include_difficult: bool,
    min_objects: int = 1,
    require_at_least_n_classes: int = 1,
    compute_cooccur: bool = True,
) -> Dict:
    ann_dir = voc_root / "Annotations"
    xml_paths = sorted(ann_dir.glob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No XML files found in {ann_dir}")

    discovered_classes = set()

    kept_images = 0
    dropped = {
        "parse_error": 0,
        "no_objects_after_filter": 0,
        "too_many_objects": 0,
        "not_enough_objects": 0,
        "not_enough_distinct_classes": 0,
    }

    per_class_object_counts = defaultdict(int)  # class -> #objects
    per_class_image_counts = defaultdict(int)   # class -> #images containing class
    objects_per_image_hist = defaultdict(int)   # n_objects -> #images

    # co-occurrence: (a,b) with a<b -> #images containing both
    cooccur = defaultdict(int)

    for xml_path in xml_paths:
        try:
            objs = parse_voc_xml_names(xml_path)
        except Exception:
            dropped["parse_error"] += 1
            continue

        # discover raw classes (before filtering)
        for o in objs:
            discovered_classes.add(o.name)

        # filter difficult if needed
        filtered = [o for o in objs if include_difficult or (not o.difficult)]

        # Part-3 style: enforce object-count limits after filtering
        n = len(filtered)
        if n == 0:
            dropped["no_objects_after_filter"] += 1
            continue
        if n < min_objects:
            dropped["not_enough_objects"] += 1
            continue
        if n > max_objects:
            dropped["too_many_objects"] += 1
            continue

        # distinct class constraint (optional)
        present_classes = sorted({o.name for o in filtered})
        if len(present_classes) < require_at_least_n_classes:
            dropped["not_enough_distinct_classes"] += 1
            continue

        # keep
        kept_images += 1
        objects_per_image_hist[n] += 1

        # counts
        for o in filtered:
            per_class_object_counts[o.name] += 1
        for c in present_classes:
            per_class_image_counts[c] += 1

        # co-occurrence
        if compute_cooccur and len(present_classes) >= 2:
            for i in range(len(present_classes)):
                for j in range(i + 1, len(present_classes)):
                    a, b = present_classes[i], present_classes[j]
                    cooccur[(a, b)] += 1

    # finalize outputs (sorted for readability)
    discovered_classes_sorted = sorted(discovered_classes)

    def sort_dict(d: Dict[str, int]) -> Dict[str, int]:
        return dict(sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))

    cooccur_sorted = None
    if compute_cooccur:
        # store as list for JSON friendliness
        cooccur_sorted = sorted(
            [{"a": a, "b": b, "images_with_both": n} for (a, b), n in cooccur.items()],
            key=lambda x: (-x["images_with_both"], x["a"], x["b"]),
        )

    stats = {
        "voc_root": str(voc_root),
        "num_total_xml": len(xml_paths),
        "max_objects": max_objects,
        "min_objects": min_objects,
        "include_difficult": include_difficult,
        "require_at_least_n_classes": require_at_least_n_classes,
        "num_discovered_classes_raw": len(discovered_classes_sorted),
        "discovered_classes_raw": discovered_classes_sorted,
        "num_kept_images_after_filtering": kept_images,
        "dropped": dropped,
        "objects_per_image_hist": dict(sorted(objects_per_image_hist.items(), key=lambda kv: kv[0])),
        "per_class_object_counts_after_filtering": sort_dict(per_class_object_counts),
        "per_class_image_counts_after_filtering": sort_dict(per_class_image_counts),
        "cooccurrence_after_filtering": cooccur_sorted,
    }
    return stats


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Part-3-style class stats for VOC (all classes).")
    p.add_argument("--voc-root", type=str, required=True, help="VOC root containing Annotations/")
    p.add_argument("--max-objects", type=int, default=3, help="Drop images with > max_objects after filtering")
    p.add_argument("--min-objects", type=int, default=1, help="Drop images with < min_objects after filtering")
    p.add_argument(
        "--require-at-least-n-classes",
        type=int,
        default=1,
        help="Drop images that contain fewer than this many distinct classes (after filtering). "
             "Set to 2 if you want multi-class frames only.",
    )
    p.add_argument("--include-difficult", action="store_true", help="Include <difficult> objects")
    p.add_argument("--no-cooccur", action="store_true", help="Skip co-occurrence computation")
    p.add_argument("--out", type=str, default="datasets/part3_allclass_stats.json", help="Output JSON path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    voc_root = find_voc_root(Path(args.voc_root))
    out_path = Path(args.out).expanduser().resolve()

    stats = compute_stats(
        voc_root=voc_root,
        max_objects=args.max_objects,
        include_difficult=args.include_difficult,
        min_objects=args.min_objects,
        require_at_least_n_classes=args.require_at_least_n_classes,
        compute_cooccur=(not args.no_cooccur),
    )

    write_json(out_path, stats)

    print(f"[VOC] root: {voc_root}")
    print(f"[CFG] max_objects={args.max_objects} min_objects={args.min_objects} "
          f"include_difficult={args.include_difficult} require>=classes={args.require_at_least_n_classes}")
    print(f"[RAW] discovered_classes={stats['num_discovered_classes_raw']}")
    print(f"[KEEP] images_after_filtering={stats['num_kept_images_after_filtering']}")
    print(f"[DROP] {stats['dropped']}")
    print(f"[OUT] wrote: {out_path}")

    # quick top-10 preview
    top10 = list(stats["per_class_object_counts_after_filtering"].items())[:10]
    print("[TOP10] per_class_object_counts_after_filtering:")
    for k, v in top10:
        print(f"  {k:>12s}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())