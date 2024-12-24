import os
import torch
from PIL import Image
import torchvision.transforms as T

def load_image(image_path, transform=None, device='cpu'):
    """
    Loads an image from disk, applies a transform, and moves it to the given device.
    """
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    image = image.unsqueeze(0).to(device)  # add batch dimension
    return image

def save_image(tensor, output_path):
    """
    Saves a torch tensor as an image file.
    Expects tensor in shape [1, 3, H, W] or [3, H, W].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    # Convert to PIL
    to_pil = T.ToPILImage()
    img = to_pil(tensor.cpu())
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, 'JPEG')

def adain_inference(
    model, 
    content_path, 
    style_path, 
    output_path, 
    content_transform, 
    style_transform, 
    alpha=1.0,
    device='cuda'
):
    """
    Runs a single pass of AdaIN style transfer using the provided model and saves the result.
    """
    content_img = load_image(content_path, content_transform, device)
    style_img   = load_image(style_path, style_transform, device)

    with torch.no_grad():
        output = model(content_img, style_img, alpha=alpha)
    
    # Optional: You can denormalize output before saving
    save_image(output, output_path)
