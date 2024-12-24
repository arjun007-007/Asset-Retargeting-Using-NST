"""
inference.py

Loads a trained (or pre-trained) AdaIN style transfer model and applies it to
new content images with selected style images.
"""

import os
import torch
import logging
from PIL import Image

from src.config import Config
from src.data.preprocessing import get_default_transforms, denormalize
from src.style_transfer.networks import VGGEncoder, Decoder, StyleTransferNet
from torchvision.utils import save_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s] [%(levelname)s] %(message)s')

def load_image(image_path, transform, device):
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)

def run_inference(content_image_path, style_image_path, output_path):
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.USE_CUDA else "cpu")

    # Initialize model
    encoder = VGGEncoder(requires_grad=False).to(device)
    decoder = Decoder().to(device)
    model   = StyleTransferNet(encoder, decoder).to(device)

    # Load decoder weights
    if not os.path.exists(cfg.INFERENCE_DECODER_PATH):
        raise FileNotFoundError(f"Inference decoder checkpoint not found at {cfg.INFERENCE_DECODER_PATH}")

    decoder.load_state_dict(torch.load(cfg.INFERENCE_DECODER_PATH, map_location=device))
    model.eval()

    # Transforms
    transform = get_default_transforms(cfg.IMAGE_SIZE)

    # Load images
    content_img = load_image(content_image_path, transform, device)
    style_img   = load_image(style_image_path, transform, device)

    with torch.no_grad():
        output = model(content_img, style_img, alpha=cfg.ALPHA)
    
    # Denormalize and save
    output = denormalize(output.squeeze(0))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(output, output_path)
    logging.info(f"Saved stylized image to {output_path}")

if __name__ == "__main__":
    # Example usage
    run_inference(
        content_image_path="data/raw/content/example_content.jpg",
        style_image_path="data/raw/style/example_style.jpg",
        output_path="results/stylized_output.jpg"
    )
