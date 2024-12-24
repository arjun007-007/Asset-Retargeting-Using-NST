import torch
from torchvision import transforms

def get_content_transform(image_size=512):
    """
    Returns a composition of transforms for content images:
    - Resize
    - CenterCrop (optional)
    - Convert to tensor
    - Normalize (using typical ImageNet stats for VGG)
    """
    transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)


def get_style_transform(image_size=512):
    """
    Returns transforms for style images, typically the same as content transforms
    so they're in the same domain. 
    """
    transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)


def denormalize_tensor(tensor):
    """
    Undo the normalization for visualization or saving.
    Args:
        tensor (torch.Tensor): Normalized image
    Returns:
        torch.Tensor: Denormalized image in [0,1] range
    """
    # If we used ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    out = tensor * std + mean
    return torch.clamp(out, 0, 1)  # ensure within [0,1]
