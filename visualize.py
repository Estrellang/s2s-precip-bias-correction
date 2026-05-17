"""
可视化脚本：生成论文所需图表
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
import torch
import xarray as xr
import cartopy.crs as ccrs
from china_map import add_china_map_base, apply_china_mask
from model import CVAE

matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['figure.dpi'] = 150

def load_test_data(data_dir="data/processed"):
    X = np.load(f"{data_dir}/X_s2s.npy").astype(np.float32)
    Y = np.load(f"{data_dir}/Y_era5.npy").astype(np.float32)
    norm = np.load(f"{data_dir}/norm_params.npy")
    X_mean, X_std, Y_mean, Y_std = norm

    n_test = int(len(X) * 0.2)
    X_test_raw = X[-n_test:]
    Y_test = Y[-n_test:]

    X_norm = (X_test_raw - X_mean) / X_std
    X_tensor = torch.tensor(X_norm[:, None])

    device = torch.device("cpu")
    model = CVAE(latent_dim=64).to(device)
    model.load_state_dict(torch.load(f"{data_dir}/best_model.pt", map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for _ in range(10):
            pred = model.predict(X_tensor).squeeze(1).numpy()
            pred = pred * Y_std + Y_mean
            preds.append(pred)
    Y_pred = np.mean(preds, axis=0)

    era5_ref = xr.open_dataset(f"{data_dir}/era5_china_summer.nc")
    lat = era5_ref["lat"].values
    lon = era5_ref["lon"].values
    era5_ref.close()

    return X_test_raw, Y_test, Y_pred, lat, lon


def plot_spatial_mean(X, Y, Y_pred, lat, lon, output_dir="figures"):
    """图1：空间分布对比（偏差图）"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    pred_mean = apply_china_mask(Y_pred.mean(0), lat, lon)
    obs_mean = apply_china_mask(Y.mean(0), lat, lon)
    s2s_bias = apply_china_mask(X.mean(0) - Y.mean(0), lat, lon)
    vmax = float(np.nanmax(np.concatenate([pred_mean.ravel(), obs_mean.ravel()])))
    bias_min = float(np.nanmin(s2s_bias))
    bias_max = float(np.nanmax(s2s_bias))
    bias_norm = colors.TwoSlopeNorm(vmin=bias_min, vcenter=0.0, vmax=bias_max)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # S2S偏差
    im0 = axes[0].imshow(
        s2s_bias,
        origin='upper',
        cmap='RdBu_r',
        norm=bias_norm,
    )
    axes[0].set_title(f'S2S偏差 (min={bias_min:.2f}, max={bias_max:.2f})')
    plt.colorbar(im0, ax=axes[0], label='mm/day')

    # CVAE订正后
    im1 = axes[1].imshow(pred_mean, origin='upper', cmap='Blues', vmin=0, vmax=vmax)
    axes[1].set_title('CVAE订正后')
    plt.colorbar(im1, ax=axes[1], label='mm/day')

    # ERA5观测
    im2 = axes[2].imshow(obs_mean, origin='upper', cmap='Blues', vmin=0, vmax=vmax)
    axes[2].set_title('ERA5观测')
    plt.colorbar(im2, ax=axes[2], label='mm/day')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/spatial_mean.png")
    plt.close()
    print("已保存: figures/spatial_mean.png")
    print(f"S2S偏差范围: {bias_min:.3f} ~ {bias_max:.3f} mm/day")


def _format_geo_axes(ax, lon, lat):
    ax.set_xlim(float(lon.min()), float(lon.max()))
    ax.set_ylim(float(lat.min()), float(lat.max()))
    ax.set_xlabel("经度 (°E)")
    ax.set_ylabel("纬度 (°N)")
    ax.set_xticks(np.arange(70, 141, 10))
    ax.set_yticks(np.arange(15, 56, 10))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def plot_spatial_mean_map(X, Y, Y_pred, lat, lon, output_dir="figures"):
    """
    地图风格空间分布图：
    以经纬度为坐标轴，用阴影图展示S2S偏差、CVAE订正和ERA5观测。
    不覆盖原始spatial_mean.png，单独输出 spatial_mean_map.png。
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    s2s_bias = apply_china_mask(X.mean(0) - Y.mean(0), lat, lon)
    pred_mean = apply_china_mask(Y_pred.mean(0), lat, lon)
    obs_mean = apply_china_mask(Y.mean(0), lat, lon)

    vmax = float(np.nanpercentile(np.concatenate([pred_mean.ravel(), obs_mean.ravel()]), 99))
    vmax = max(vmax, 1.0)
    bias_lim = float(np.nanpercentile(np.abs(s2s_bias), 98))
    bias_lim = max(bias_lim, 1.0)

    lon2d, lat2d = np.meshgrid(lon, lat)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True, sharey=True)

    im0 = axes[0].pcolormesh(
        lon2d,
        lat2d,
        s2s_bias,
        shading="auto",
        cmap="RdBu_r",
        vmin=-bias_lim,
        vmax=bias_lim,
    )
    axes[0].set_title("S2S偏差场 (S2S - ERA5)")
    _format_geo_axes(axes[0], lon, lat)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, label="mm/day")

    im1 = axes[1].pcolormesh(
        lon2d,
        lat2d,
        pred_mean,
        shading="auto",
        cmap="Blues",
        vmin=0.0,
        vmax=vmax,
    )
    axes[1].set_title("CVAE订正后平均场")
    _format_geo_axes(axes[1], lon, lat)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, label="mm/day")

    im2 = axes[2].pcolormesh(
        lon2d,
        lat2d,
        obs_mean,
        shading="auto",
        cmap="Blues",
        vmin=0.0,
        vmax=vmax,
    )
    axes[2].set_title("ERA5观测平均场")
    _format_geo_axes(axes[2], lon, lat)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, label="mm/day")

    plt.suptitle("中国境内夏季降水空间分布（经纬度底图风格）", y=0.98)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spatial_mean_map.png")
    plt.close()
    print("已保存: figures/spatial_mean_map.png")


def _add_cartopy_base(ax, extent):
    add_china_map_base(ax, extent, draw_left=True, draw_bottom=True)


def plot_spatial_mean_cartopy(X, Y, Y_pred, lat, lon, output_dir="figures"):
    """
    Cartopy地图版空间分布图：
    使用经纬度投影、海岸线和国界作为底图。
    单独输出 spatial_mean_cartopy.png，不覆盖其他图件。
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    s2s_bias = apply_china_mask(X.mean(0) - Y.mean(0), lat, lon)
    pred_mean = apply_china_mask(Y_pred.mean(0), lat, lon)
    obs_mean = apply_china_mask(Y.mean(0), lat, lon)

    vmax = float(np.nanpercentile(np.concatenate([pred_mean.ravel(), obs_mean.ravel()]), 99))
    vmax = max(vmax, 1.0)
    bias_lim = float(np.nanpercentile(np.abs(s2s_bias), 98))
    bias_lim = max(bias_lim, 1.0)
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    lon2d, lat2d = np.meshgrid(lon, lat)
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.4),
        subplot_kw={"projection": proj},
    )

    for ax in axes:
        _add_cartopy_base(ax, extent)

    im0 = axes[0].pcolormesh(
        lon2d,
        lat2d,
        s2s_bias,
        transform=proj,
        shading="auto",
        cmap="RdBu_r",
        vmin=-bias_lim,
        vmax=bias_lim,
        zorder=1,
    )
    axes[0].set_title("(a) S2S偏差场", fontsize=11)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, label="mm/day")

    im1 = axes[1].pcolormesh(
        lon2d,
        lat2d,
        pred_mean,
        transform=proj,
        shading="auto",
        cmap="Blues",
        vmin=0.0,
        vmax=vmax,
        zorder=1,
    )
    axes[1].set_title("(b) CVAE订正后平均场", fontsize=11)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, label="mm/day")

    im2 = axes[2].pcolormesh(
        lon2d,
        lat2d,
        obs_mean,
        transform=proj,
        shading="auto",
        cmap="Blues",
        vmin=0.0,
        vmax=vmax,
        zorder=1,
    )
    axes[2].set_title("(c) ERA5观测平均场", fontsize=11)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, label="mm/day")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spatial_mean_cartopy.png")
    plt.close()
    print("已保存: figures/spatial_mean_cartopy.png")


def plot_scatter(Y, Y_pred, output_dir="figures"):
    """图2：散点图（订正后 vs 观测）"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(Y.flatten(), Y_pred.flatten(), alpha=0.1, s=1)
    lim = max(Y.max(), Y_pred.max())
    ax.plot([0, lim], [0, lim], 'r--', label='1:1线')
    corr = np.corrcoef(Y.flatten(), Y_pred.flatten())[0, 1]
    ax.set_xlabel('ERA5观测 (mm/day)')
    ax.set_ylabel('CVAE订正后 (mm/day)')
    ax.set_title(f'散点图 (r={corr:.3f})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scatter.png")
    plt.close()
    print("已保存: figures/scatter.png")


def plot_time_series(Y, Y_pred, X, n=100, output_dir="figures"):
    """图3：区域平均降水样本序列对比（双子图）"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    obs_series = Y[:n].mean((1, 2))
    cvae_series = Y_pred[:n].mean((1, 2))
    s2s_series = X[:n].mean((1, 2))

    ymin = min(obs_series.min(), cvae_series.min(), s2s_series.min())
    ymax = max(obs_series.max(), cvae_series.max(), s2s_series.max())
    pad = 0.08 * (ymax - ymin)
    ylims = (ymin - pad, ymax + pad)

    # 上图：CVAE vs ERA5
    ax1.plot(obs_series, label='ERA5观测', color='black', linewidth=1.5)
    ax1.plot(cvae_series, label='CVAE订正后', color='blue', alpha=0.8)
    ax1.set_ylabel('降水 (mm/day)')
    ax1.set_title('CVAE订正效果对比')
    ax1.set_ylim(*ylims)
    ax1.legend()

    # 下图：S2S原始 vs ERA5（线性坐标）
    ax2.plot(s2s_series, color='red', alpha=0.7, label='S2S原始')
    ax2.plot(obs_series, color='black', linewidth=1.5, label='ERA5观测')
    ax2.set_ylabel('降水 (mm/day)')
    ax2.set_xlabel('样本序号')
    ax2.set_title('S2S原始预报 vs ERA5')
    ax2.set_ylim(*ylims)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_series.png")
    plt.close()
    print("已保存: figures/time_series.png")


if __name__ == "__main__":
    X, Y, Y_pred, lat, lon = load_test_data()
    plot_spatial_mean(X, Y, Y_pred, lat, lon)
    plot_spatial_mean_map(X, Y, Y_pred, lat, lon)
    plot_spatial_mean_cartopy(X, Y, Y_pred, lat, lon)
    plot_scatter(Y, Y_pred)
    plot_time_series(Y, Y_pred, X)
    print("所有图表生成完成！")
