import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        def down(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        def up(in_c, out_c, dropout=False):
            layers = [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # -------------------------
        # Encoder
        # -------------------------
        self.down1 = down(1, 64, normalize=False)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)

        # -------------------------
        # Decoder
        # -------------------------
        self.up1 = up(512, 256, dropout=True)
        self.up2 = up(512, 128, dropout=True)
        self.up3 = up(256, 64)

        # 🔥 FIXED FINAL LAYER (NO TRANSPOSE CONV)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)   # 64
        d2 = self.down2(d1)  # 128
        d3 = self.down3(d2)  # 256
        d4 = self.down4(d3)  # 512

        # Decoder + skip connections
        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))

        output = self.final(torch.cat([u3, d1], dim=1))

        return output