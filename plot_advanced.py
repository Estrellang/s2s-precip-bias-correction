"""
生成参考文献风格的补充图表
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import xarray as xr
from model import CVAE
from scipy.stats import pearsonr

matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['figure.dpi'] = 150

def load_sample_years():
    """按对齐脚本的真实规则恢复每个配对样本对应的有效年份。"""
    s2s = xr.open_dataset('data/processed/s2s_china_summer.nc', decode_timedelta=False)
    era5 = xr.open_dataset('data/processed/era5_china_summer.nc')

    era5_dates = {str(t)[:10] for t in era5.time.values}
    sample_years = []

    for t in s2s.time.values:
        for step in s2s.step.values:
            valid_time = t + np.timedelta64(int(step), 'ns')
            valid_date = str(valid_time)[:10]
            if valid_date in era5_dates:
                sample_years.append(int(valid_date[:4]))

    s2s.close()
    era5.close()
    return np.array(sample_years, dtype=np.int32)

def load_data_by_year():
    """按年份加载数据"""
    X = np.load('data/processed/X_s2s.npy').astype(np.float32)
    Y = np.load('data/processed/Y_era5.npy').astype(np.float32)
    norm = np.load('data/processed/norm_params.npy')
    X_mean, X_std, Y_mean, Y_std = norm

    # 加载模型
    device = torch.device("cpu")
    model = CVAE(latent_dim=64).to(device)
    model.load_state_dict(torch.load('data/processed/best_model.pt', map_location=device))
    model.eval()

    # 预测
    X_norm = (X - X_mean) / X_std
    X_tensor = torch.tensor(X_norm[:, None])
    torch.manual_seed(0)
    with torch.no_grad():
        preds = []
        for _ in range(10):
            pred = model.predict(X_tensor).squeeze(1).numpy()
            pred = pred * Y_std + Y_mean
            preds.append(pred)
    Y_pred = np.mean(preds, axis=0)

    sample_years = load_sample_years()
    if len(sample_years) != len(X):
        raise ValueError(
            f"样本年份数量({len(sample_years)})与配对样本数({len(X)})不一致，"
            "请检查年份恢复逻辑。"
        )

    yearly_data = {}
    for year in sorted(np.unique(sample_years)):
        mask = sample_years == year
        yearly_data[year] = {
            'X': X[mask],
            'Y': Y[mask],
            'Y_pred': Y_pred[mask]
        }

    return yearly_data

def plot_radar_chart(output_dir='figures'):
    """图1：雷达图 - 逐年相关系数对比"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    yearly_data = load_data_by_year()
    years = sorted(yearly_data.keys())

    # 计算逐年相关系数
    s2s_corr = []
    cvae_corr = []
    for year in years:
        data = yearly_data[year]
        # S2S vs ERA5
        r_s2s, _ = pearsonr(data['X'].flatten(), data['Y'].flatten())
        # CVAE vs ERA5
        r_cvae, _ = pearsonr(data['Y_pred'].flatten(), data['Y'].flatten())
        s2s_corr.append(float(r_s2s))
        cvae_corr.append(float(r_cvae))

    # 绘制雷达图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2 * np.pi, len(years), endpoint=False).tolist()
    s2s_corr += s2s_corr[:1]
    cvae_corr += cvae_corr[:1]
    angles += angles[:1]
    rmax = max(max(s2s_corr[:-1]), max(cvae_corr[:-1]))
    upper = min(1.0, max(0.8, rmax + 0.05))

    # S2S原始
    ax1.plot(angles, s2s_corr, 'o-', linewidth=2, color='red', label='S2S')
    ax1.fill(angles, s2s_corr, alpha=0.25, color='red')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(years)
    ax1.set_ylim(0, upper)
    ax1.set_title('S2S原始预报\nmean={:.4f}'.format(np.mean(s2s_corr[:-1])),
                  fontsize=14, pad=20)
    ax1.grid(True)

    # CVAE订正
    ax2.plot(angles, cvae_corr, 'o-', linewidth=2, color='blue', label='CVAE')
    ax2.fill(angles, cvae_corr, alpha=0.25, color='blue')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(years)
    ax2.set_ylim(0, upper)
    ax2.set_title('CVAE订正后\nmean={:.4f}'.format(np.mean(cvae_corr[:-1])),
                  fontsize=14, pad=20)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/radar_yearly_correlation.png', bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_dir}/radar_yearly_correlation.png")

if __name__ == "__main__":
    plot_radar_chart()
