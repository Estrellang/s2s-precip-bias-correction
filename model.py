"""
CVAE模型定义
基于卷积网络的条件变分自编码器
"""

import torch
import torch.nn as nn

class Encoder(nn.Module):
    """编码器：输入(S2S预报 + ERA5观测) -> 潜在空间(mu, logvar)"""

    def __init__(self, latent_dim=64):
        super().__init__()
        # 输入通道=2（S2S + ERA5拼接）
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        # 33x57 -> 9x15 after two stride-2 convs
        self.flatten_dim = 128 * 9 * 15
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x, condition):
        # x: S2S预报, condition: ERA5观测
        inp = torch.cat([x, condition], dim=1)  # (B, 2, H, W)
        h = self.conv(inp).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """解码器：潜在变量 + S2S条件 -> 订正后降水"""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.flatten_dim = 128 * 9 * 15
        self.fc = nn.Linear(latent_dim + 33 * 57, self.flatten_dim)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, z, condition):
        cond_flat = condition.flatten(1)
        h = torch.relu(self.fc(torch.cat([z, cond_flat], dim=1)))
        h = h.view(-1, 128, 9, 15)
        return self.deconv(h)[:, :, :33, :57]


class CVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, x), mu, logvar

    def predict(self, x):
        z = torch.randn(x.size(0), self.latent_dim).to(x.device)
        return self.decoder(z, x)