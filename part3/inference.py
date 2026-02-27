from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from part3.model import FixedSlotDetector, ModelConfig


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


def load_classes(classes_json: str | Path) -> Tuple[List[str], int]:
    obj = json.loads(Path(classes_json).read_text(encoding="utf-8"))
    classes = obj["classes"]
    bg_id = int(obj.get("bg_id", len(classes) - 1))
    return classes, bg_id


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> List[int]:
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    pil_img: Image.Image,
    classes: List[str],
    bg_id: int,
    image_size: int,
    conf_thresh: float,
    nms_iou: float,
    device: torch.device,
):
    img = pil_img.convert("RGB").resize((image_size, image_size))
    x = TF.to_tensor(img).unsqueeze(0).to(device)

    out = model(x)
    boxes = out["boxes"][0].cpu().numpy()          # [K,4]
    logits = out["logits"][0].cpu()
    probs = torch.softmax(logits, dim=-1).numpy()  # [K,C]

    dets = []
    for k in range(boxes.shape[0]):
        p = probs[k].copy()
        p[bg_id] = -1.0
        cls = int(np.argmax(p))
        score = float(probs[k, cls])
        if score < conf_thresh:
            continue
        dets.append((cls, score, boxes[k]))

    # class-wise NMS
    final = []
    for cls in sorted(set([d[0] for d in dets])):
        cls_dets = [d for d in dets if d[0] == cls]
        b = np.stack([d[2] for d in cls_dets], axis=0)
        s = np.array([d[1] for d in cls_dets], dtype=np.float32)
        keep = nms_xyxy(b, s, iou_thresh=nms_iou)
        for i in keep:
            final.append(cls_dets[i])

    # sort by score
    final.sort(key=lambda x: x[1], reverse=True)
    return final, img


def draw_dets_cv2(img_bgr: np.ndarray, dets, classes: List[str]):
    h, w = img_bgr.shape[:2]
    for cls, score, box in dets:
        x1, y1, x2, y2 = [int(v) for v in box]
        color = VOC_COLORS[cls % len(VOC_COLORS)]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{classes[cls]} {score:.2f}"
        cv2.putText(img_bgr, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img_bgr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--classes-json", type=str, required=True)
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--folder", type=str, default=None)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--out", type=str, default="outputs/part3/infer")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--conf-thresh", type=float, default=0.25)
    p.add_argument("--nms-iou", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device()
    classes, bg_id = load_classes(args.classes_json)

    cfg = ModelConfig(
        image_size=args.image_size,
        max_objects=3,
        num_classes_total=len(classes),
        bg_id=bg_id,
        backbone="mobilenet_v3_small",
        pretrained=False,
    )
    model = FixedSlotDetector(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        pil = Image.open(args.image)
        dets, img = predict_image(model, pil, classes, bg_id, args.image_size, args.conf_thresh, args.nms_iou, device)
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        bgr = draw_dets_cv2(bgr, dets, classes)
        out_path = out_dir / "image_out.jpg"
        cv2.imwrite(str(out_path), bgr)
        print(f"[OK] wrote {out_path}")
        return

    if args.folder:
        folder = Path(args.folder)
        imgs = sorted([p for p in folder.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        for p in imgs[:200]:
            pil = Image.open(p)
            dets, img = predict_image(model, pil, classes, bg_id, args.image_size, args.conf_thresh, args.nms_iou, device)
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            bgr = draw_dets_cv2(bgr, dets, classes)
            out_path = out_dir / f"{p.stem}_out.jpg"
            cv2.imwrite(str(out_path), bgr)
        print(f"[OK] wrote {len(imgs[:200])} images to {out_dir}")
        return

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = out_dir / "video_out.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (args.image_size, args.image_size))

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            dets, img = predict_image(model, pil, classes, bg_id, args.image_size, args.conf_thresh, args.nms_iou, device)
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            bgr = draw_dets_cv2(bgr, dets, classes)
            writer.write(bgr)

        cap.release()
        writer.release()
        print(f"[OK] wrote {out_path}")
        return

    print("Provide --image or --folder or --video")


if __name__ == "__main__":
    main()