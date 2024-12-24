"""
train.py

Trains/fine-tunes an AdaIN-based style transfer model using a folder of content images
and a folder of style images.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

from src.config import Config
from src.data.dataset import TextureDataset
from src.data.preprocessing import get_default_transforms
from src.style_transfer.networks import VGGEncoder, Decoder, StyleTransferNet, calc_mean_std

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def train_adain():
    cfg = Config()

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.USE_CUDA else "cpu")
    logging.info(f"Using device: {device}")

    # Create dataset & dataloaders
    content_dataset = TextureDataset(
        root_dir=cfg.CONTENT_DIR,
        transform=get_default_transforms(cfg.IMAGE_SIZE)
    )
    style_dataset = TextureDataset(
        root_dir=cfg.STYLE_DIR,
        transform=get_default_transforms(cfg.IMAGE_SIZE)
    )

    content_loader = DataLoader(content_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    style_loader   = DataLoader(style_dataset,   batch_size=cfg.BATCH_SIZE, shuffle=True)

    # Initialize models
    encoder = VGGEncoder(requires_grad=False)
    decoder = Decoder()
    model   = StyleTransferNet(encoder, decoder).to(device)

    # Optionally load existing decoder weights
    if cfg.PRETRAINED_DECODER_PATH and os.path.exists(cfg.PRETRAINED_DECODER_PATH):
        logging.info(f"Loading pretrained decoder from {cfg.PRETRAINED_DECODER_PATH}")
        decoder.load_state_dict(torch.load(cfg.PRETRAINED_DECODER_PATH, map_location=device))
    
    # Define optimizer for the decoder (since encoder is fixed)
    optimizer = optim.Adam(decoder.parameters(), lr=cfg.LEARNING_RATE)

    # Loss functions
    mse_loss = nn.MSELoss()

    # We will train the decoder to reconstruct style-transferred images
    logging.info(f"Starting training for {cfg.EPOCHS} epochs.")

    for epoch in range(cfg.EPOCHS):
        model.train()  # put in training mode
        total_content_loss = 0.0
        total_style_loss   = 0.0
        
        # Zip content and style loaders. This means we'll iterate in parallel.
        for (content_imgs, _), (style_imgs, _) in zip(content_loader, style_loader):
            content_imgs = content_imgs.to(device)
            style_imgs   = style_imgs.to(device)

            # Forward: encode content and style, apply AdaIN
            content_features = model.encoder(content_imgs)
            style_features   = model.encoder(style_imgs)
            t = model.adain(content_features, style_features)

            # alpha blend
            t = cfg.ALPHA * t + (1 - cfg.ALPHA) * content_features

            # Decode
            generated = model.decoder(t)

            # Re-encode the generated image
            generated_features = model.encoder(generated)

            # Compute content loss
            content_loss = mse_loss(generated_features, t)

            # Compute style loss with means and std
            g_mean, g_std = calc_mean_std(generated_features)
            s_mean, s_std = calc_mean_std(style_features)
            style_loss    = mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)

            # Total loss
            loss = cfg.CONTENT_WEIGHT * content_loss + cfg.STYLE_WEIGHT * style_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_content_loss += content_loss.item()
            total_style_loss   += style_loss.item()

        # Epoch summary
        avg_content_loss = total_content_loss / len(content_loader)
        avg_style_loss   = total_style_loss / len(content_loader)
        logging.info(f"Epoch [{epoch+1}/{cfg.EPOCHS}] "
                     f"Content Loss: {avg_content_loss:.4f}, "
                     f"Style Loss: {avg_style_loss:.4f}")

        # Optional: save model checkpoint each epoch
        if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f"decoder_epoch_{epoch+1}.pth")
            torch.save(decoder.state_dict(), checkpoint_path)
            logging.info(f"Saved checkpoint: {checkpoint_path}")

    logging.info("Training complete.")

if __name__ == "__main__":
    train_adain()
