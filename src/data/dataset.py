"""
dataset.py

Defines a PyTorch Dataset class for loading texture images from a directory.
"""

import os
import logging
from PIL import Image
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)

class TextureDataset(Dataset):
    """
    A PyTorch Dataset for loading texture images from a given root directory.
    This dataset scans all subfolders for image files.

    Args:
        root_dir (str): Path to the root directory containing texture images.
        transform (callable, optional): A function/transform to apply to the PIL image.
    """
    def __init__(self, root_dir, transform=None):
        super(TextureDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform

        # Collect all image file paths
        self.image_paths = []
        for subdir, _, files in os.walk(self.root_dir):
            for file_name in files:
                if is_image_file(file_name):
                    full_path = os.path.join(subdir, file_name)
                    self.image_paths.append(full_path)

        if not self.image_paths:
            logging.warning(f"No image files found in {root_dir}!")
        else:
            logging.info(f"Found {len(self.image_paths)} images in {root_dir}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Transformed image tensor
            path (str): Path to the original image file
        """
        image_path = self.image_paths[idx]
        # Load with PIL
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {str(e)}")
            # Optionally return a dummy image or raise an exception
            raise e

        if self.transform:
            image = self.transform(image)

        return image, image_path
