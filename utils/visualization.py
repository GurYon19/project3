"""
Visualization utilities for Object Detection.
"""
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
from typing import List, Tuple, Optional


# Color palette for different classes
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128)
]


def draw_box(image: Image.Image, box: List[float], color: Tuple = (255, 0, 0),
             label: str = None, thickness: int = 2) -> Image.Image:
    """
    Draw bounding box on image.
    Box format: [x_center, y_center, w, h] normalized to [0, 1]
    """
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size
    
    x_center, y_center, w, h = box
    x1 = int((x_center - w / 2) * img_w)
    y1 = int((y_center - h / 2) * img_h)
    x2 = int((x_center + w / 2) * img_w)
    y2 = int((y_center + h / 2) * img_h)
    
    draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
    
    if label:
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
                      fill=color)
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
    
    return image


def visualize_predictions(image: Image.Image, pred_box: List[float], 
                         target_box: List[float] = None, iou: float = None) -> Image.Image:
    """
    Visualize prediction and ground truth boxes.
    """
    result = image.copy()
    
    # Draw prediction (green)
    result = draw_box(result, pred_box, color=(0, 255, 0), label="Pred")
    
    # Draw ground truth (red)
    if target_box:
        result = draw_box(result, target_box, color=(255, 0, 0), label="GT")
    
    # Add IoU text
    if iou is not None:
        draw = ImageDraw.Draw(result)
        draw.text((10, 10), f"IoU: {iou:.3f}", fill=(255, 255, 255))
    
    return result


def visualize_multi_predictions(image: Image.Image, predictions: dict,
                                targets: dict = None, class_names: List[str] = None) -> Image.Image:
    """
    Visualize multi-object predictions.
    """
    result = image.copy()
    
    pred_boxes = predictions['boxes']
    pred_classes = predictions.get('classes', None)
    pred_obj = predictions.get('objectness', None)
    
    for i, box in enumerate(pred_boxes):
        if pred_obj is not None and pred_obj[i] < 0.5:
            continue
        
        color = COLORS[i % len(COLORS)]
        label = None
        
        if pred_classes is not None and class_names:
            if isinstance(pred_classes[i], torch.Tensor):
                class_idx = pred_classes[i].argmax().item()
            else:
                class_idx = int(pred_classes[i])
            label = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        
        result = draw_box(result, box.tolist() if isinstance(box, torch.Tensor) else box,
                         color=color, label=label)
    
    return result


def create_video_from_frames(frames: List[np.ndarray], output_path: str, fps: int = 30):
    """
    Create video from list of frames.
    """
    if len(frames) == 0:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()


def run_video_inference(model, video_path: str, output_path: str, transform,
                       device: str = 'cpu', class_names: List[str] = None):
    """
    Run inference on video and save result.
    """
    model.eval()
    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        if isinstance(output, dict):
            result_image = visualize_multi_predictions(pil_image, {
                'boxes': output['boxes'][0].cpu(),
                'classes': output['classes'][0].cpu() if 'classes' in output else None,
                'objectness': output['objectness'][0].cpu() if 'objectness' in output else None
            }, class_names=class_names)
        else:
            pred_box = output[0].cpu().tolist()
            result_image = visualize_predictions(pil_image, pred_box)
        
        result_frame = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        result_frame = cv2.resize(result_frame, (width, height))
        out.write(result_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    print(f"Video saved to: {output_path}")
