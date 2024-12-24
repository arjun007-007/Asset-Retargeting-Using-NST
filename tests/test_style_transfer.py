"""
test_style_transfer.py

A basic test to ensure the style transfer pipeline can process one content+style pair
without crashing and produces a valid output tensor.
"""

import os
import unittest
import torch
from PIL import Image

from src.config import Config
from src.style_transfer.networks import VGGEncoder, Decoder, StyleTransferNet
from src.data.preprocessing import get_default_transforms

class TestStyleTransfer(unittest.TestCase):

    def setUp(self):
        # Setup config & model
        self.cfg = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.USE_CUDA else "cpu")

        self.encoder = VGGEncoder(requires_grad=False).to(self.device)
        self.decoder = Decoder().to(self.device)
        self.model   = StyleTransferNet(self.encoder, self.decoder).to(self.device)

        # If we have a decoder checkpoint, load it
        if os.path.exists(self.cfg.INFERENCE_DECODER_PATH):
            self.decoder.load_state_dict(torch.load(self.cfg.INFERENCE_DECODER_PATH, map_location=self.device))
        
        self.model.eval()

        # Transforms
        self.transform = get_default_transforms(self.cfg.IMAGE_SIZE)

        # Example content & style
        self.content_path = os.path.join(self.cfg.CONTENT_DIR, "example_content.jpg")
        self.style_path   = os.path.join(self.cfg.STYLE_DIR, "example_style.jpg")

        if not os.path.exists(self.content_path) or not os.path.exists(self.style_path):
            self.skipTest("Test images not found. Skipping style transfer test.")

    def test_style_transfer_inference(self):
        # Load images
        content_img = Image.open(self.content_path).convert('RGB')
        style_img   = Image.open(self.style_path).convert('RGB')

        content_tensor = self.transform(content_img).unsqueeze(0).to(self.device)
        style_tensor   = self.transform(style_img).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(content_tensor, style_tensor, alpha=self.cfg.ALPHA)
        
        self.assertEqual(output.shape, content_tensor.shape, "Output should match the dimensions of the input.")
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN values!")

if __name__ == '__main__':
    unittest.main()
