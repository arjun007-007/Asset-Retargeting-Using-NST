"""
test_data_loading.py

Tests for verifying that the Dataset and dataloader work properly.
"""

import os
import unittest
from torch.utils.data import DataLoader

from src.data.dataset import TextureDataset
from src.data.preprocessing import get_default_transforms

class TestDataLoading(unittest.TestCase):

    def test_texture_dataset(self):
        # Assuming we have at least one image in data/raw/content/
        content_dir = "data/raw/content"
        if not os.path.exists(content_dir):
            self.skipTest(f"Directory {content_dir} does not exist.")
        
        dataset = TextureDataset(root_dir=content_dir, transform=get_default_transforms(256))
        self.assertGreater(len(dataset), 0, "Dataset should have at least one image.")

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for i, (img, path) in enumerate(dataloader):
            self.assertEqual(img.ndim, 4, "Image should have batch dimension.")
            self.assertEqual(img.shape[1], 3, "Image should have 3 channels (RGB).")
            break  # Just test the first batch

if __name__ == '__main__':
    unittest.main()
