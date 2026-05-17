"""
误差分布分析脚本
补充区域尺度和代表点位的误差分布统计，用于论文结果扩展。
"""

import argparse
import csv
import os

# 避免在受限环境下matplotlib缓存目录不可写
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import numpy as np
import torch
import xarray as xr

from model import CVAE


def setup_chinese_font():
    """配置中文字体，减少绘图中文乱码。"""
    candidates = [
        "Arial Unicode MS",
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "SimHei",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available:
            matplotlib.rcParams["font.family"] = font_name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


REGION_DEFS = [
    ("西北", 35.0, 50.0, 75.0, 100.0),
    ("华北", 35.0, 43.0, 110.0, 122.0),
    ("西南", 22.0, 33.0, 95.0, 108.0),
    ("江淮", 28.0, 35.0, 108.0, 122.0),
    ("华南", 20.0, 28.0, 105.0, 120.0),
]

POINT_DEFS = [
    ("北京", 39.90, 116.40),
    ("上海", 31.23, 121.47),
    ("广州", 23.13, 113.27),
    ("成都", 30.67, 104.06),
    ("乌鲁木齐", 43.82, 87.62),
    ("哈尔滨", 45.75, 126.63),
]


def load_test_predictions(data_dir="data/processed", n_ens=10):
    """加载测试集并生成CVAE预测。"""
    X = np.load(f"{data_dir}/X_s2s.npy").astype(np.float32)
    Y = np.load(f"{data_dir}/Y_era5.npy").astype(np.float32)
    norm = np.load(f"{data_dir}/norm_params.npy")
    X_mean, X_std, Y_mean, Y_std = norm

    n_test = int(len(X) * 0.2)
    X_test = X[-n_test:]
    Y_test = Y[-n_test:]

    X_test_norm = (X_test - X_mean) / X_std
    X_tensor = torch.tensor(X_test_norm[:, None])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(latent_dim=64).to(device)
    model.load_state_dict(torch.load(f"{data_dir}/best_model.pt", map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        for _ in range(n_ens):
            pred = model.predict(X_tensor).squeeze(1).cpu().numpy()
            pred = pred * Y_std + Y_mean
            preds.append(pred)
    Y_pred = np.mean(preds, axis=0)

    return X_test, Y_test, Y_pred


def nearest_index(values, target):
    """返回离target最近的索引。"""
    return int(np.argmin(np.abs(values - target)))


def calc_stats(err):
    """误差分布统计量。"""
    return {
        "bias": float(np.mean(err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "std": float(np.std(err)),
        "median": float(np.median(err)),
        "p05": float(np.percentile(err, 5)),
        "p25": float(np.percentile(err, 25)),
        "p75": float(np.percentile(err, 75)),
        "p95": float(np.percentile(err, 95)),
        "positive_ratio": float(np.mean(err > 0) * 100.0),
    }


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_region_boxplot(region_plot_data, output_path):
    """区域误差箱线图（S2S vs CVAE）。"""
    fig, ax = plt.subplots(figsize=(14, 6))

    all_data = []
    positions = []
    colors = []
    xticks = []
    xticklabels = []

    for i, item in enumerate(region_plot_data):
        pos_s2s = i * 3 + 1
        pos_cvae = i * 3 + 2
        all_data.extend([item["s2s_err"], item["cvae_err"]])
        positions.extend([pos_s2s, pos_cvae])
        colors.extend(["tab:red", "tab:blue"])
        xticks.append((pos_s2s + pos_cvae) / 2)
        xticklabels.append(item["name"])

    bp = ax.boxplot(
        all_data,
        positions=positions,
        widths=0.7,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("误差 (mm/day)")
    ax.set_title("分区域误差分布对比（S2S vs CVAE）")
    ax.legend(
        [
            plt.Line2D([0], [0], color="tab:red", lw=8, alpha=0.45),
            plt.Line2D([0], [0], color="tab:blue", lw=8, alpha=0.45),
        ],
        ["S2S原始误差", "CVAE订正误差"],
        loc="upper right",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_point_hist(point_plot_data, output_path):
    """代表点位误差直方图。"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes = axes.flatten()

    for ax, item in zip(axes, point_plot_data):
        s2s_err = item["s2s_err"]
        cvae_err = item["cvae_err"]
        vals = np.concatenate([s2s_err, cvae_err])
        x_max = np.percentile(np.abs(vals), 99)
        x_max = max(x_max, 0.5)
        bins = np.linspace(-x_max, x_max, 45)

        ax.hist(np.clip(s2s_err, -x_max, x_max), bins=bins, density=True,
                alpha=0.45, color="tab:red", label="S2S")
        ax.hist(np.clip(cvae_err, -x_max, x_max), bins=bins, density=True,
                alpha=0.45, color="tab:blue", label="CVAE")
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{item['name']} ({item['lat']:.2f}N, {item['lon']:.2f}E)")
        ax.set_xlabel("误差 (mm/day)")

    axes[0].set_ylabel("概率密度")
    axes[0].legend(loc="upper right")
    plt.suptitle("代表点位误差分布（为可读性截断到99%分位）", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def write_summary(summary_path, region_rows, point_rows):
    """导出可直接放进论文的简短文字结论。"""
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("误差分布补充结论（自动生成）\n")
        f.write("=" * 40 + "\n")

        if region_rows:
            best_region = max(region_rows, key=lambda x: x["rmse_improve_%"])
            worst_region = min(region_rows, key=lambda x: x["rmse_improve_%"])
            f.write(
                f"区域层面：RMSE改善最明显的是{best_region['region']} "
                f"({best_region['rmse_improve_%']:.2f}%)；"
                f"改善最弱的是{worst_region['region']} "
                f"({worst_region['rmse_improve_%']:.2f}%)。\n"
            )

        if point_rows:
            best_point = max(point_rows, key=lambda x: x["rmse_improve_%"])
            worst_point = min(point_rows, key=lambda x: x["rmse_improve_%"])
            f.write(
                f"点位层面：RMSE改善最明显的是{best_point['point']} "
                f"({best_point['rmse_improve_%']:.2f}%)；"
                f"改善最弱的是{worst_point['point']} "
                f"({worst_point['rmse_improve_%']:.2f}%)。\n"
            )

        f.write("详细统计请见region_error_stats.csv和point_error_stats.csv。\n")


def analyze(data_dir="data/processed", output_dir="figures"):
    setup_chinese_font()
    os.makedirs(output_dir, exist_ok=True)

    X_test, Y_test, Y_pred = load_test_predictions(data_dir=data_dir)
    s2s_err = X_test - Y_test
    cvae_err = Y_pred - Y_test

    grid_ds = xr.open_dataset(f"{data_dir}/era5_china_summer.nc")
    lat = grid_ds["lat"].values
    lon = grid_ds["lon"].values
    grid_ds.close()
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    region_rows = []
    region_plot_data = []
    for name, lat_min, lat_max, lon_min, lon_max in REGION_DEFS:
        mask = (
            (lat2d >= lat_min) & (lat2d <= lat_max) &
            (lon2d >= lon_min) & (lon2d <= lon_max)
        )
        if not np.any(mask):
            continue

        region_s2s = s2s_err[:, mask].reshape(-1)
        region_cvae = cvae_err[:, mask].reshape(-1)
        s2s_stats = calc_stats(region_s2s)
        cvae_stats = calc_stats(region_cvae)
        rmse_improve = (s2s_stats["rmse"] - cvae_stats["rmse"]) / (s2s_stats["rmse"] + 1e-8) * 100
        mae_improve = (s2s_stats["mae"] - cvae_stats["mae"]) / (s2s_stats["mae"] + 1e-8) * 100

        region_rows.append({
            "region": name,
            "grid_cells": int(mask.sum()),
            "n_error_samples": int(region_s2s.size),
            "bias_s2s": s2s_stats["bias"],
            "bias_cvae": cvae_stats["bias"],
            "mae_s2s": s2s_stats["mae"],
            "mae_cvae": cvae_stats["mae"],
            "rmse_s2s": s2s_stats["rmse"],
            "rmse_cvae": cvae_stats["rmse"],
            "rmse_improve_%": rmse_improve,
            "mae_improve_%": mae_improve,
            "std_s2s": s2s_stats["std"],
            "std_cvae": cvae_stats["std"],
            "median_s2s": s2s_stats["median"],
            "median_cvae": cvae_stats["median"],
            "p05_s2s": s2s_stats["p05"],
            "p05_cvae": cvae_stats["p05"],
            "p25_s2s": s2s_stats["p25"],
            "p25_cvae": cvae_stats["p25"],
            "p75_s2s": s2s_stats["p75"],
            "p75_cvae": cvae_stats["p75"],
            "p95_s2s": s2s_stats["p95"],
            "p95_cvae": cvae_stats["p95"],
            "positive_ratio_s2s_%": s2s_stats["positive_ratio"],
            "positive_ratio_cvae_%": cvae_stats["positive_ratio"],
        })
        region_plot_data.append({
            "name": name,
            "s2s_err": region_s2s,
            "cvae_err": region_cvae,
        })

    point_rows = []
    point_plot_data = []
    for name, plat, plon in POINT_DEFS:
        i = nearest_index(lat, plat)
        j = nearest_index(lon, plon)

        point_s2s = s2s_err[:, i, j]
        point_cvae = cvae_err[:, i, j]
        s2s_stats = calc_stats(point_s2s)
        cvae_stats = calc_stats(point_cvae)
        rmse_improve = (s2s_stats["rmse"] - cvae_stats["rmse"]) / (s2s_stats["rmse"] + 1e-8) * 100
        mae_improve = (s2s_stats["mae"] - cvae_stats["mae"]) / (s2s_stats["mae"] + 1e-8) * 100

        point_rows.append({
            "point": name,
            "target_lat": plat,
            "target_lon": plon,
            "grid_lat": float(lat[i]),
            "grid_lon": float(lon[j]),
            "n_error_samples": int(point_s2s.size),
            "bias_s2s": s2s_stats["bias"],
            "bias_cvae": cvae_stats["bias"],
            "mae_s2s": s2s_stats["mae"],
            "mae_cvae": cvae_stats["mae"],
            "rmse_s2s": s2s_stats["rmse"],
            "rmse_cvae": cvae_stats["rmse"],
            "rmse_improve_%": rmse_improve,
            "mae_improve_%": mae_improve,
            "std_s2s": s2s_stats["std"],
            "std_cvae": cvae_stats["std"],
            "median_s2s": s2s_stats["median"],
            "median_cvae": cvae_stats["median"],
            "p05_s2s": s2s_stats["p05"],
            "p05_cvae": cvae_stats["p05"],
            "p95_s2s": s2s_stats["p95"],
            "p95_cvae": cvae_stats["p95"],
            "positive_ratio_s2s_%": s2s_stats["positive_ratio"],
            "positive_ratio_cvae_%": cvae_stats["positive_ratio"],
        })
        point_plot_data.append({
            "name": name,
            "lat": float(lat[i]),
            "lon": float(lon[j]),
            "s2s_err": point_s2s,
            "cvae_err": point_cvae,
        })

    region_csv = os.path.join(output_dir, "region_error_stats.csv")
    point_csv = os.path.join(output_dir, "point_error_stats.csv")
    region_png = os.path.join(output_dir, "region_error_boxplot.png")
    point_png = os.path.join(output_dir, "point_error_hist.png")
    summary_txt = os.path.join(output_dir, "error_distribution_summary.txt")

    save_csv(region_csv, region_rows)
    save_csv(point_csv, point_rows)
    plot_region_boxplot(region_plot_data, region_png)
    plot_point_hist(point_plot_data, point_png)
    write_summary(summary_txt, region_rows, point_rows)

    print(f"已保存: {region_csv}")
    print(f"已保存: {point_csv}")
    print(f"已保存: {region_png}")
    print(f"已保存: {point_png}")
    print(f"已保存: {summary_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="区域/点位误差分布分析")
    parser.add_argument("--data_dir", default="data/processed", help="数据目录")
    parser.add_argument("--output_dir", default="figures", help="输出目录")
    args = parser.parse_args()
    analyze(data_dir=args.data_dir, output_dir=args.output_dir)
