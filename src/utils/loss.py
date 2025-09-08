import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


# reference: https://github.com/dxyang/StyleTransfer/blob/master/vgg.py
class PerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vgg_features = vgg16(weights="IMAGENET1K_V1").features

        self.to_relu_1_2 = nn.Sequential(*list(vgg_features[:4]))
        self.to_relu_2_2 = nn.Sequential(*list(vgg_features[4:9]))
        self.to_relu_3_3 = nn.Sequential(*list(vgg_features[9:16]))
        self.to_relu_4_3 = nn.Sequential(*list(vgg_features[16:23]))

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            h_relu_1_2: (B, 64, H, W)
            h_relu_2_2: (B, 128, H/2, W/2)
            h_relu_3_3: (B, 256, H/4, W/4)
            h_relu_4_3: (B, 512, H/8, W/8)
        """
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        return h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            y: (B, C, H, W)
        Returns:
            A scalar tensor of the perceptual loss
        """
        x_features = self.forward_features(x)
        y_features = self.forward_features(y)
        loss = 0.0
        for hx, hy in zip(x_features, y_features):
            loss += F.mse_loss(hx, hy)
        return loss


if __name__ == "__main__":
    loss = PerceptualLoss()
    rand_img_x = torch.randn(16, 3, 224, 224)
    rand_img_y = torch.randn(16, 3, 224, 224)
    print(loss(rand_img_x, rand_img_y))
