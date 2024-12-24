"""
preprocessing.py

Contains utility functions and transform pipelines for texture preprocessing.
"""

import torch
from torchvision import transforms

def get_default_transforms(image_size=256):
    """
    Returns a default transform pipeline:
      - Resize to image_size x image_size
      - Convert to torch.Tensor
      - Normalize using ImageNet mean/std (commonly used for style transfer with VGG)
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_augmentation_transforms(image_size=256):
    """
    Returns a transform pipeline with data augmentation.
    Example usage in training to help the model generalize better.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # Example augmentations:
        # transforms.RandomRotation(15),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def denormalize(tensor):
    """
    Reverts the normalization for visualization or saving.
    Assumes ImageNet mean/std.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    return torch.clamp(tensor * std + mean, 0, 1)
