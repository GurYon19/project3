
import json
import sys

def check_json(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"File: {path}")
        print(f"Images: {len(data['images'])}")
        print(f"Annotations: {len(data['annotations'])}")
        print(f"Categories: {len(data['categories'])}")
    except Exception as e:
        print(f"Error reading {path}: {e}")

if __name__ == "__main__":
    check_json("datasets/part2/train/_annotations.coco.json")
