"""
模型评估脚本
计算RMSE、相关系数、偏差等指标
"""

import torch
import numpy as np
from model import CVAE

def evaluate(data_dir="data/processed"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据和标准化参数
    X = np.load(f"{data_dir}/X_s2s.npy").astype(np.float32)
    Y = np.load(f"{data_dir}/Y_era5.npy").astype(np.float32)
    norm = np.load(f"{data_dir}/norm_params.npy")
    X_mean, X_std, Y_mean, Y_std = norm

    # 标准化
    X_norm = (X - X_mean) / X_std
    Y_norm = (Y - Y_mean) / Y_std

    # 使用测试集（后20%）
    n = len(X)
    n_test = int(n * 0.2)
    X_test = torch.tensor(X_norm[-n_test:, None]).to(device)
    Y_test = Y[-n_test:]  # 原始单位

    # 加载模型
    model = CVAE(latent_dim=64).to(device)
    model.load_state_dict(torch.load(f"{data_dir}/best_model.pt", map_location=device))
    model.eval()

    # 预测（多次采样取均值）
    preds = []
    with torch.no_grad():
        for _ in range(10):
            pred = model.predict(X_test).squeeze(1).cpu().numpy()
            pred = pred * Y_std + Y_mean  # 反标准化
            preds.append(pred)
    Y_pred = np.mean(preds, axis=0)

    # 计算指标
    rmse = np.sqrt(np.mean((Y_pred - Y_test) ** 2))
    corr = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0, 1]
    bias = np.mean(Y_pred - Y_test)
    # X这里已经是原始单位(mm/day)，无需再做反标准化
    s2s_rmse = np.sqrt(np.mean((X[-n_test:] - Y_test) ** 2))

    print(f"{'='*40}")
    print(f"原始S2S RMSE:   {s2s_rmse:.4f} mm/day")
    print(f"CVAE订正 RMSE:  {rmse:.4f} mm/day")
    print(f"相关系数:        {corr:.4f}")
    print(f"偏差:            {bias:.4f} mm/day")
    print(f"RMSE改善:        {(s2s_rmse - rmse) / s2s_rmse * 100:.1f}%")
    print(f"{'='*40}")

    return Y_pred, Y_test, X[-n_test:]

if __name__ == "__main__":
    evaluate()
