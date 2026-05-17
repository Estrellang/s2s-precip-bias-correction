"""
CVAE训练脚本
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import CVAE

def cvae_loss(recon, y, mu, logvar):
    """重构损失 + KL散度"""
    recon_loss = nn.MSELoss()(recon.squeeze(1), y)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.001 * kl_loss

def train(data_dir="data/processed", epochs=100, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    X = np.load(f"{data_dir}/X_s2s.npy").astype(np.float32)
    Y = np.load(f"{data_dir}/Y_era5.npy").astype(np.float32)

    # 标准化
    X_mean, X_std = X.mean(), X.std() + 1e-8
    Y_mean, Y_std = Y.mean(), Y.std() + 1e-8
    X = (X - X_mean) / X_std
    Y = (Y - Y_mean) / Y_std

    # 保存标准化参数
    np.save(f"{data_dir}/norm_params.npy",
            np.array([X_mean, X_std, Y_mean, Y_std]))

    # 添加通道维度
    X_t = torch.tensor(X[:, None])  # (N, 1, H, W)
    Y_t = torch.tensor(Y[:, None])

    # 划分数据集（训练/验证/测试 = 7:1:2）
    n = len(X_t)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    dataset = TensorDataset(X_t, Y_t)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n - n_train - n_val]
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = CVAE(latent_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            recon, mu, logvar = model(x, y)
            loss = cvae_loss(recon, y.squeeze(1), mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                recon, mu, logvar = model(x, y)
                val_loss += cvae_loss(recon, y.squeeze(1), mu, logvar).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "data/processed/best_model.pt")

    print(f"训练完成！最优验证损失: {best_val_loss:.4f}")
    return model

if __name__ == "__main__":
    train()
