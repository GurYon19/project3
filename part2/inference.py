"""
Live Inference Script for Part 2: Single Object Detection
Runs detection on images/video and displays results in real-time.
"""
import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEVICE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from models.detector import create_detector


def load_model(checkpoint_path: str):
    """Load trained model from checkpoint."""
    model = create_detector(phase=2, config={'pretrained': False, 'freeze_backbone': False})
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print(f"✅ Loaded model from: {checkpoint_path}")
    print(f"   Best IoU: {checkpoint.get('best_metric', 'N/A'):.4f}")
    return model


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Preprocess image for model input."""
    # Resize
    img_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] then apply ImageNet normalization
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_normalized = (img_normalized - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    
    # Convert to tensor [C, H, W]
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
    
    return img_tensor.unsqueeze(0)  # Add batch dimension


def cxwh_to_xyxy(box, img_w, img_h):
    """Convert normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]."""
    cx, cy, w, h = box
    x1 = int((cx - w/2) * img_w)
    y1 = int((cy - h/2) * img_h)
    x2 = int((cx + w/2) * img_w)
    y2 = int((cy + h/2) * img_h)
    return x1, y1, x2, y2


def draw_detection(image: np.ndarray, box: list, label: str = "Tiger", color=(0, 255, 0)):
    """Draw bounding box and label on image."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = cxwh_to_xyxy(box, w, h)
    
    # Draw box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    
    # Draw label background
    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y1_label = max(y1, label_size[1])
    cv2.rectangle(image, (x1, y1_label - label_size[1] - 10), 
                  (x1 + label_size[0], y1_label), color, -1)
    
    # Draw label text
    cv2.putText(image, label, (x1, y1_label - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return image


def run_inference_on_image(model, image_path: str):
    """Run inference on a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    # Preprocess and predict
    input_tensor = preprocess_image(image).to(DEVICE)
    
    with torch.no_grad():
        pred_box = model(input_tensor).squeeze(0).cpu().numpy()
    
    # Draw result
    result = draw_detection(image.copy(), pred_box)
    
    # Show
    cv2.imshow("Detection Result", result)
    print(f"Prediction: cx={pred_box[0]:.3f}, cy={pred_box[1]:.3f}, w={pred_box[2]:.3f}, h={pred_box[3]:.3f}")
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_inference_on_folder(model, folder_path: str, output_folder: str = None):
    """Run inference on all images in a folder."""
    folder = Path(folder_path)
    output = Path(output_folder) if output_folder else folder / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = set()
    for ext in extensions:
        image_paths.update(folder.glob(ext))
    image_paths = list(image_paths)
    
    print(f"Found {len(image_paths)} images")
    
    for i, img_path in enumerate(image_paths):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Predict
        input_tensor = preprocess_image(image).to(DEVICE)
        with torch.no_grad():
            pred_box = model(input_tensor).squeeze(0).cpu().numpy()
        
        # Draw and save
        result = draw_detection(image.copy(), pred_box)
        out_path = output / f"pred_{img_path.name}"
        cv2.imwrite(str(out_path), result)
        
        print(f"[{i+1}/{len(image_paths)}] Saved: {out_path.name}")
    
    print(f"\n✅ Predictions saved to: {output}")


def run_inference_on_video(model, video_path: str, output_path: str = None):
    """Run inference on video file."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup output
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict
            input_tensor = preprocess_image(frame).to(DEVICE)
            with torch.no_grad():
                pred_box = model(input_tensor).squeeze(0).cpu().numpy()
            
            # Draw
            result = draw_detection(frame, pred_box)
            
            # Add frame counter
            cv2.putText(result, f"Frame: {frame_count}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show
            cv2.imshow("Live Detection", result)
            
            # Write to output
            if writer:
                writer.write(result)
            
            frame_count += 1
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    print(f"\n✅ Processed {frame_count} frames")
    if output_path:
        print(f"✅ Output saved to: {output_path}")


def run_webcam_inference(model):
    """Run live inference on webcam."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("🎥 Webcam inference started. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict
            input_tensor = preprocess_image(frame).to(DEVICE)
            with torch.no_grad():
                pred_box = model(input_tensor).squeeze(0).cpu().numpy()
            
            # Draw
            result = draw_detection(frame, pred_box)
            
            # Show FPS
            cv2.putText(result, "Press 'q' to quit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Live Detection (Webcam)", result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run object detection inference")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/part2/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None, help='Single image path')
    parser.add_argument('--folder', type=str, default=None, help='Folder of images')
    parser.add_argument('--video', type=str, default=None, help='Video file path')
    parser.add_argument('--webcam', action='store_true', help='Run on webcam')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint)
    
    # Run inference based on input type
    if args.webcam:
        run_webcam_inference(model)
    elif args.video:
        run_inference_on_video(model, args.video, args.output)
    elif args.folder:
        run_inference_on_folder(model, args.folder, args.output)
    elif args.image:
        run_inference_on_image(model, args.image)
    else:
        # Default: run on validation set
        print("No input specified. Running on validation set...")
        run_inference_on_folder(model, 'datasets/part2/valid', 'outputs/inference_results')
