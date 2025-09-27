# augmentation/dataset_manager.py

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

class DatasetManager:
    """
    Manages all file system interactions for the dataset augmentation process.
    Handles reading, writing, and directory creation.
    """
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def setup_directories(self, splits: List[str] = ['train', 'val'], augmented_split: str = 'train_augmented'):
        """Creates the necessary output directory structure."""
        for split in splits:
            (self.output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        if augmented_split:
            (self.output_path / augmented_split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / augmented_split / 'labels').mkdir(parents=True, exist_ok=True)
        
        print("Output directories created successfully.")

    def get_data_files(self, split: str) -> List[Tuple[Path, Path]]:
        """
        Gets a list of (image_path, label_path) tuples for a given split.
        Handles flexible input directory structures.
        """
        split_img_path, split_lbl_path = None, None
        possible_paths = [
            (self.input_path / split / 'images', self.input_path / split / 'labels'),
            (self.input_path / 'images' / split, self.input_path / 'labels' / split),
        ]

        for img_path, lbl_path in possible_paths:
            if img_path.exists() and lbl_path.exists():
                split_img_path, split_lbl_path = img_path, lbl_path
                break
        
        if not split_img_path:
            print(f"Warning: Could not find data for split '{split}' in any expected location.")
            return []

        image_files = list(split_img_path.glob('*.jpg')) + list(split_img_path.glob('*.png'))
        data_files = []
        for img_file in image_files:
            label_file = split_lbl_path / f"{img_file.stem}.txt"
            if label_file.exists():
                data_files.append((img_file, label_file))
        
        return data_files

    def load_data_item(self, image_path: Path, label_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Loads a single image and its corresponding labels."""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return None
        
        try:
            labels = np.loadtxt(str(label_path)).reshape(-1, 5)
            return image, labels
        except Exception as e:
            print(f"Warning: Could not read or parse label file {label_path}: {e}")
            return None

    def save_data_item(self, image: np.ndarray, labels: np.ndarray, split: str, filename: str):
        """Saves an image and its labels to a specific split directory."""
        output_img_path = self.output_path / split / 'images' / filename
        output_lbl_path = self.output_path / split / 'labels' / f"{Path(filename).stem}.txt"
        
        cv2.imwrite(str(output_img_path), image)
        np.savetxt(str(output_lbl_path), labels, fmt='%d %.6f %.6f %.6f %.6f')

    def save_augmented_item(self, image: np.ndarray, labels: np.ndarray, original_filename: str, 
                            aug_type: str, aug_index: int, augmented_split: str = 'train_augmented'):
        """Saves an augmented item with a prefixed filename."""
        original_path = Path(original_filename)
        prefix = f"{aug_type}_{aug_index:04d}"
        new_filename = f"{prefix}_{original_path.stem}{original_path.suffix}"
        
        self.save_data_item(image, labels, augmented_split, new_filename)