"""
Download sample images for Part 1 classification demo.
Downloads real photos from public sources (Unsplash, Wikimedia Commons).
"""
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO

# Create output directory
output_dir = Path("part1_images")
output_dir.mkdir(exist_ok=True)

# Sample images from Wikimedia Commons (public domain)
images = [
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/800px-Cat03.jpg",
        "name": "cat.jpg",
        "description": "Tabby Cat"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/800px-YellowLabradorLooking_new.jpg",
        "name": "dog.jpg",
        "description": "Golden Labrador"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Domestic_goat_kid_in_capeweed.jpg/800px-Domestic_goat_kid_in_capeweed.jpg",
        "name": "goat.jpg",
        "description": "Goat"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Square_200x200.png/800px-Square_200x200.png",
        "name": "bird.jpg",
        "description": "Bird"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/800px-Collage_of_Nine_Dogs.jpg",
        "name": "dogs.jpg",
        "description": "Multiple Dogs"
    }
]

# Alternative: Use placeholder images if Wikimedia fails
placeholder_images = [
    {
        "url": "https://picsum.photos/800/600?random=1",
        "name": "sample1.jpg",
        "description": "Sample 1"
    },
    {
        "url": "https://picsum.photos/800/600?random=2",
        "name": "sample2.jpg",
        "description": "Sample 2"
    },
    {
        "url": "https://picsum.photos/800/600?random=3",
        "name": "sample3.jpg",
        "description": "Sample 3"
    },
    {
        "url": "https://picsum.photos/800/600?random=4",
        "name": "sample4.jpg",
        "description": "Sample 4"
    },
    {
        "url": "https://picsum.photos/800/600?random=5",
        "name": "sample5.jpg",
        "description": "Sample 5"
    }
]

def download_image(url, save_path, description):
    """Download and save an image."""
    try:
        print(f"Downloading: {description}...")
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        # Open and convert to RGB
        img = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Save
        img.save(save_path)
        print(f"  ✓ Saved to: {save_path}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def main():
    print("="*60)
    print("Downloading Sample Images for Part 1 Classification")
    print("="*60)
    print()
    
    success_count = 0
    
    # Try Wikimedia Commons images first
    for img_info in images:
        save_path = output_dir / img_info["name"]
        if download_image(img_info["url"], save_path, img_info["description"]):
            success_count += 1
    
    # If we didn't get enough images, try placeholders
    if success_count < 3:
        print("\nTrying alternative sources...")
        for img_info in placeholder_images[:5-success_count]:
            save_path = output_dir / img_info["name"]
            download_image(img_info["url"], save_path, img_info["description"])
    
    print()
    print("="*60)
    print(f"Download complete! {success_count} images saved to '{output_dir}'")
    print("="*60)
    print()
    print("Next step: Run the classification demo")
    print("  python part1_classification.py")

if __name__ == "__main__":
    main()
