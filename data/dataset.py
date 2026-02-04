"""
Dataset classes for Object Detection.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Callable


class SingleObjectDataset(Dataset):
    """Dataset for single-object detection (Part 2)."""
    
    def __init__(self, images_dir: str, annotations_file: str = None, 
                 transform: Callable = None, annotation_format: str = 'coco'):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.annotation_format = annotation_format
        
        if annotation_format == 'coco' and annotations_file:
            self.samples = self._load_coco_annotations(annotations_file)
        elif annotation_format == 'yolo':
            self.samples = self._load_yolo_annotations()
        else:
            self.samples = self._load_images_only()
    
    def _load_coco_annotations(self, annotations_file: str) -> List[Dict]:
        with open(annotations_file, 'r') as f:
            coco = json.load(f)
        
        id_to_image = {img['id']: img for img in coco['images']}
        image_to_anno = {}
        for anno in coco['annotations']:
            img_id = anno['image_id']
            if img_id not in image_to_anno:
                image_to_anno[img_id] = []
            image_to_anno[img_id].append(anno)
        
        samples = []
        for img_id, annos in image_to_anno.items():
            if len(annos) != 1:
                continue
            
            img_info = id_to_image[img_id]
            anno = annos[0]
            x, y, w, h = anno['bbox']
            img_w, img_h = img_info['width'], img_info['height']
            
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            
            samples.append({
                'image_path': self.images_dir / img_info['file_name'],
                'box': [x_center, y_center, norm_w, norm_h]
            })
        
        return samples
    
    def _load_yolo_annotations(self) -> List[Dict]:
        samples = []
        for img_path in self.images_dir.glob('*.jpg'):
            txt_path = img_path.with_suffix('.txt')
            if not txt_path.exists():
                continue
            
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) != 1:
                continue
            
            parts = lines[0].strip().split()
            x_center, y_center, w, h = map(float, parts[1:5])
            
            samples.append({
                'image_path': img_path,
                'box': [x_center, y_center, w, h]
            })
        
        return samples
    
    def _load_images_only(self) -> List[Dict]:
        samples = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in self.images_dir.glob(ext):
                samples.append({'image_path': img_path, 'box': None})
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        
        box = sample['box']
        if box is not None:
            box = torch.tensor(box, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            image, box = self.transform(image, box)
        
        if box is not None:
            box = box.squeeze(0)
        else:
            box = torch.zeros(4)
        
        return image, box


class MultiObjectDataset(Dataset):
    """Dataset for multi-object detection (Part 3)."""
    
    def __init__(self, images_dir: str, annotations_file: str, class_names: List[str],
                 transform: Callable = None, max_objects: int = 3, annotation_format: str = 'coco'):
        self.images_dir = Path(images_dir)
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.transform = transform
        self.max_objects = max_objects
        
        if annotation_format == 'coco':
            self.samples = self._load_coco_annotations(annotations_file)
        else:
            self.samples = self._load_voc_annotations(annotations_file)
    
    def _load_coco_annotations(self, annotations_file: str) -> List[Dict]:
        with open(annotations_file, 'r') as f:
            coco = json.load(f)
        
        cat_id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}
        id_to_image = {img['id']: img for img in coco['images']}
        
        image_to_annos = {}
        for anno in coco['annotations']:
            img_id = anno['image_id']
            if img_id not in image_to_annos:
                image_to_annos[img_id] = []
            image_to_annos[img_id].append(anno)
        
        samples = []
        for img_id, annos in image_to_annos.items():
            if not (1 <= len(annos) <= self.max_objects):
                continue
            
            img_info = id_to_image[img_id]
            img_w, img_h = img_info['width'], img_info['height']
            
            boxes = []
            classes = []
            
            for anno in annos[:self.max_objects]:
                x, y, w, h = anno['bbox']
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                boxes.append([x_center, y_center, norm_w, norm_h])
                
                cat_name = cat_id_to_name.get(anno['category_id'], 'unknown')
                classes.append(self.class_to_idx.get(cat_name, 0))
            
            samples.append({
                'image_path': self.images_dir / img_info['file_name'],
                'boxes': boxes,
                'classes': classes
            })
        
        return samples
    
    def _load_voc_annotations(self, annotations_dir: str) -> List[Dict]:
        samples = []
        annotations_path = Path(annotations_dir)
        
        for xml_path in annotations_path.glob('*.xml'):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            filename = root.find('filename').text
            size = root.find('size')
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)
            
            objects = root.findall('object')
            if not (1 <= len(objects) <= self.max_objects):
                continue
            
            boxes = []
            classes = []
            
            for obj in objects[:self.max_objects]:
                name = obj.find('name').text.lower()
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                x_center = ((xmin + xmax) / 2) / img_w
                y_center = ((ymin + ymax) / 2) / img_h
                w = (xmax - xmin) / img_w
                h = (ymax - ymin) / img_h
                
                boxes.append([x_center, y_center, w, h])
                classes.append(self.class_to_idx.get(name, 0))
            
            samples.append({
                'image_path': self.images_dir / filename,
                'boxes': boxes,
                'classes': classes
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        
        boxes = torch.tensor(sample['boxes'], dtype=torch.float32)
        classes = torch.tensor(sample['classes'], dtype=torch.long)
        
        if self.transform:
            image, boxes = self.transform(image, boxes)
        
        num_objects = boxes.size(0)
        valid_mask = torch.zeros(self.max_objects)
        valid_mask[:num_objects] = 1
        
        padded_boxes = torch.zeros(self.max_objects, 4)
        padded_boxes[:num_objects] = boxes
        
        padded_classes = torch.zeros(self.max_objects, dtype=torch.long)
        padded_classes[:num_objects] = classes
        
        return image, {'boxes': padded_boxes, 'classes': padded_classes, 'valid_mask': valid_mask}


def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
