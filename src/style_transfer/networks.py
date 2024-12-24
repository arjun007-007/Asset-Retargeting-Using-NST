import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGGEncoder(nn.Module):
    """
    A VGG-19 based encoder that extracts content features (conv4_1, by default).
    """
    def __init__(self, requires_grad=False):
        super(VGGEncoder, self).__init__()
        vgg_pretrained = models.vgg19(pretrained=True).features
        
        # We'll slice up to conv4_1: indices up to 21 (PyTorch VGG19)
        self.slice1 = nn.Sequential(*[vgg_pretrained[i] for i in range(0, 2)])   # relu1_1
        self.slice2 = nn.Sequential(*[vgg_pretrained[i] for i in range(2, 7)])   # relu2_1
        self.slice3 = nn.Sequential(*[vgg_pretrained[i] for i in range(7, 12)])  # relu3_1
        self.slice4 = nn.Sequential(*[vgg_pretrained[i] for i in range(12, 21)]) # relu4_1
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Returns the output after the fourth slice (relu4_1),
        which we typically use for content representation.
        """
        x = self.slice1(x)
        x = self.slice2(x)
        x = self.slice3(x)
        x = self.slice4(x)
        return x


class Decoder(nn.Module):
    """
    A decoder network that mirrors the VGG19 up to conv4_1 in reverse,
    but with learnable parameters to reconstruct an image from features.
    """
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            # Input will be features from relu4_1
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # upsample
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)


class StyleTransferNet(nn.Module):
    """
    Combines VGGEncoder, AdaIN layer, and Decoder into one pipeline.
    """
    def __init__(self, encoder, decoder):
        super(StyleTransferNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def adain(self, content_features, style_features, eps=1e-5):
        """
        Adaptive Instance Normalization (AdaIN).
        """
        c_mean, c_std = calc_mean_std(content_features, eps)
        s_mean, s_std = calc_mean_std(style_features, eps)
        
        normalized = (content_features - c_mean) / (c_std + eps) * s_std + s_mean
        return normalized

    def forward(self, content, style, alpha=1.0):
        """
        content: content image tensor
        style: style image tensor
        alpha: interpolation factor between content and style
        """
        # Extract features
        content_feat = self.encoder(content)
        style_feat   = self.encoder(style)

        # Perform AdaIN
        t = self.adain(content_feat, style_feat)
        # Optional: blend between content features and style-transferred features
        t = alpha * t + (1 - alpha) * content_feat

        # Decode to image space
        return self.decoder(t)


def calc_mean_std(feat, eps=1e-5):
    """
    feat: (N, C, H, W)
    """
    N, C = feat.size(0), feat.size(1)
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
