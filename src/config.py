"""
config.py

Contains configuration constants and hyperparameters for style transfer training and inference.
"""

import os

class Config:
    # Directories
    DATA_DIR           = "data"
    CONTENT_DIR        = os.path.join(DATA_DIR, "raw", "content")
    STYLE_DIR          = os.path.join(DATA_DIR, "raw", "style")
    CHECKPOINT_DIR     = "models"

    # Training hyperparameters
    EPOCHS             = 5
    BATCH_SIZE         = 2
    IMAGE_SIZE         = 256
    LEARNING_RATE      = 1e-4
    CONTENT_WEIGHT     = 1.0
    STYLE_WEIGHT       = 10.0
    ALPHA              = 1.0  # blending factor between content and style features

    # Hardware
    USE_CUDA           = True  # if True, use GPU if available

    # Pretrained model paths
    PRETRAINED_DECODER_PATH  = "models/decoder.pth" 
    INFERENCE_DECODER_PATH   = "models/decoder_epoch_5.pth"

    # Saving frequency
    SAVE_INTERVAL      = 1  # save checkpoint every N epochs
