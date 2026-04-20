import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c, stride=2, normalize=True):
            layers = [
                nn.Conv2d(in_c, out_c, 4, stride, 1)
            ]

            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))  # 🔥 better than BatchNorm

            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return nn.Sequential(*layers)

        # -------------------------
        # PatchGAN discriminator
        # -------------------------
        self.model = nn.Sequential(

            # Input: gray (1) + color (3) = 4 channels
            block(4, 64, stride=2, normalize=False),  # (B,64,64,64)

            block(64, 128, stride=2),                # (B,128,32,32)
            block(128, 256, stride=2),               # (B,256,16,16)

            # smaller stride → more patch resolution
            block(256, 512, stride=1),               # (B,512,15,15)

            # 🔥 Final Patch Output
            nn.Conv2d(512, 1, 4, stride=1, padding=1)  # (B,1,14,14)
        )

    def forward(self, gray, color):
        # Concatenate grayscale + color
        x = torch.cat([gray, color], dim=1)

        return self.model(x)