"""
Part 1: Classification Inference Demo
Runs inference on sample images using pretrained MobileNetV3-Small.
"""
import torch
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.backbone import get_classification_model, get_imagenet_labels
from data.transforms import get_inference_transform


def load_local_images(image_dir="part1_images"):
    """
    Load sample images from local directory for classification demo.
    Returns list of (image, description) tuples.
    """
    # Get image directory path
    images_path = Path(__file__).parent / image_dir
    
    if not images_path.exists():
        print(f"Error: Image directory '{images_path}' not found!")
        return []
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    images = []
    for img_file in sorted(images_path.iterdir()):
        if img_file.suffix.lower() in image_extensions:
            try:
                img = Image.open(img_file).convert('RGB')
                # Use filename without extension as description
                desc = img_file.stem.replace('_', ' ').title()
                images.append((img, desc))
                print(f"Loaded: {desc} ({img_file.name})")
            except Exception as e:
                print(f"Failed to load {img_file.name}: {e}")
    
    return images


def run_classification(model, transform, images, labels, device='cpu', top_k=5):
    """
    Run classification on images and return results.
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for img, desc in images:
            # Preprocess
            input_tensor = transform(img).unsqueeze(0).to(device)
            
            # Inference
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Top-K predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            predictions = []
            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                predictions.append({
                    'class': labels[idx],
                    'probability': float(prob) * 100
                })
            
            results.append({
                'description': desc,
                'image': img,
                'predictions': predictions
            })
    
    return results


def display_results(results, save_path=None):
    """
    Display classification results with images in two 2x3 grids (12 total).
    """
    n = len(results)
    
    # Create figure with 2 rows of grids
    fig = plt.figure(figsize=(12, 10))
    
    # Grid 1: Top 6 images (2 rows × 3 columns)
    for idx in range(6):
        ax = plt.subplot(4, 3, idx + 1)  # Positions 1-6 in top half
        
        if idx < n:
            result = results[idx]
            ax.imshow(result['image'])
            ax.axis('off')
            
            # Build title
            title = f"Ground Truth: {result['description']}\n"
            title += "-" * 25 + "\n"
            title += "Predictions:\n"
            for i, pred in enumerate(result['predictions'][:2]):
                title += f"{i+1}. {pred['class']}: {pred['probability']:.1f}%\n"
            ax.set_title(title, fontsize=8, ha='center')
        else:
            ax.axis('off')
    
    # Grid 2: Bottom 6 images (2 rows × 3 columns)
    for idx in range(6, 12):
        ax = plt.subplot(4, 3, idx + 1)  # Positions 7-12 in bottom half
        
        if idx < n:
            result = results[idx]
            ax.imshow(result['image'])
            ax.axis('off')
            
            # Build title
            title = f"Ground Truth: {result['description']}\n"
            title += "-" * 25 + "\n"
            title += "Predictions:\n"
            for i, pred in enumerate(result['predictions'][:2]):
                title += f"{i+1}. {pred['class']}: {pred['probability']:.1f}%\n"
            ax.set_title(title, fontsize=8, ha='center')
        else:
            ax.axis('off')
    
    plt.tight_layout(pad=1.5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Results saved to: {save_path}")
    
    plt.show()






def print_results_table(results):
    """Print results as a formatted table."""
    print("\n" + "="*80)
    print("CLASSIFICATION RESULTS - MobileNetV3-Small")
    print("="*80)
    
    for result in results:
        print(f"\n>>> Image: {result['description']}")
        print("-" * 40)
        for i, pred in enumerate(result['predictions']):
            print(f"  {i+1}. {pred['class']:30s} {pred['probability']:6.2f}%")


def main():
    """Main function for Part 1 classification demo."""
    print("="*60)
    print("Part 1: MobileNetV3-Small Classification Demo")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\nLoading pretrained MobileNetV3-Small...")
    model = get_classification_model(pretrained=True)
    model = model.to(device)
    print("Model loaded successfully!")
    
    # Get ImageNet labels
    labels = get_imagenet_labels()
    print(f"Loaded {len(labels)} ImageNet classes")
    
    # Transform
    transform = get_inference_transform(image_size=224)
    
    # Load sample images from local directory
    print("\nLoading sample images from 'part1_images' directory...")
    images = load_local_images()
    
    if not images:
        print("No images found. Please add images to the 'part1_images' directory.")
        return
    
    # Run classification
    print("\nRunning inference...")
    results = run_classification(model, transform, images, labels, device)
    
    # Print results
    print_results_table(results)
    
    # Display with visualization
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "part1_classification_results.png"
    
    display_results(results, save_path=str(save_path))
    
    print("\n" + "="*60)
    print("Part 1 Classification Demo Complete!")
    print("="*60)
    
    # Model info for report
    print("\n>>> MODEL ARCHITECTURE INFO FOR REPORT:")
    print(f"  - Model: MobileNetV3-Small")
    print(f"  - Input size: 224x224")
    print(f"  - Output classes: 1000 (ImageNet)")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")


if __name__ == "__main__":
    main()
