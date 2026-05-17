"""
Leadtime技巧评估脚本
按指定lead day评估每个网格点的ACC和RMSE，并对比S2S原始与CVAE订正。
"""

import argparse
import csv
import os

# 避免受限环境中matplotlib缓存目录不可写
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LogNorm
import numpy as np
import torch
import xarray as xr
import cartopy.crs as ccrs

from china_map import add_china_map_base, apply_china_mask
from model import CVAE


def setup_chinese_font():
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


def parse_lead_days(text):
    days = []
    for x in text.split(","):
        x = x.strip()
        if x:
            days.append(int(x))
    return sorted(set(days))


def load_model_and_norm(data_dir):
    norm = np.load(f"{data_dir}/norm_params.npy")
    x_mean, x_std, y_mean, y_std = norm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(latent_dim=64).to(device)
    model.load_state_dict(torch.load(f"{data_dir}/best_model.pt", map_location=device))
    model.eval()
    return model, device, x_mean, x_std, y_mean, y_std


def collect_pairs_by_lead(s2s_path, era5_path, target_leads):
    """
    为每个lead day收集配对样本：
    X_lead: S2S原始预报
    Y_lead: ERA5观测
    """
    s2s = xr.open_dataset(s2s_path, decode_timedelta=False)
    era5 = xr.open_dataset(era5_path)

    # 与align_data.py保持一致：按日期字符串对齐
    era5_dates = {str(t)[:10]: t for t in era5.time.values}
    lead_days_all = np.array([int(step / 1e9 / 3600 / 24) for step in s2s.step.values], dtype=int)

    results = {}
    for lead in target_leads:
        step_idx = np.where(lead_days_all == lead)[0]
        if len(step_idx) == 0:
            results[lead] = {"X": None, "Y": None, "n": 0}
            continue

        x_list, y_list = [], []
        j = int(step_idx[0])  # 每个lead day仅对应一个step
        step_ns = int(s2s.step.values[j])

        for i, t in enumerate(s2s.time.values):
            valid_time = t + np.timedelta64(step_ns, "ns")
            valid_date = str(valid_time)[:10]
            if valid_date in era5_dates:
                x = s2s["tp"].isel(time=i, step=j).values
                y = era5["tp"].sel(time=era5_dates[valid_date]).values
                x = np.nan_to_num(x, nan=0.0)
                y = np.nan_to_num(y, nan=0.0)
                x_list.append(x)
                y_list.append(y)

        if x_list:
            X = np.asarray(x_list, dtype=np.float32)
            Y = np.asarray(y_list, dtype=np.float32)
            results[lead] = {"X": X, "Y": Y, "n": len(X)}
        else:
            results[lead] = {"X": None, "Y": None, "n": 0}

    lat = s2s["latitude"].values
    lon = s2s["longitude"].values
    s2s.close()
    era5.close()
    return results, lat, lon, np.unique(lead_days_all)


def predict_cvae(model, device, x, x_mean, x_std, y_mean, y_std, n_ens=10):
    """对单个lead的X进行CVAE预测（集合均值）。"""
    x_norm = (x - x_mean) / x_std
    x_tensor = torch.tensor(x_norm[:, None]).to(device)

    preds = []
    with torch.no_grad():
        for _ in range(n_ens):
            pred = model.predict(x_tensor).squeeze(1).cpu().numpy()
            pred = pred * y_std + y_mean
            preds.append(pred)
    return np.mean(preds, axis=0)


def rmse_map(pred, obs):
    return np.sqrt(np.mean((pred - obs) ** 2, axis=0))


def acc_map(pred, obs):
    """
    每个网格点上的ACC（时间维相关系数）：
    ACC = corr(pred_anom, obs_anom)，anom按该点样本均值去除。
    """
    pred_anom = pred - pred.mean(axis=0, keepdims=True)
    obs_anom = obs - obs.mean(axis=0, keepdims=True)

    numerator = np.sum(pred_anom * obs_anom, axis=0)
    denom = np.sqrt(np.sum(pred_anom ** 2, axis=0) * np.sum(obs_anom ** 2, axis=0))
    acc = np.divide(numerator, denom, out=np.full_like(numerator, np.nan), where=denom > 1e-12)
    return np.clip(acc, -1.0, 1.0)


def _add_cartopy_base(ax, extent, draw_left=True, draw_bottom=True):
    add_china_map_base(ax, extent, draw_left=draw_left, draw_bottom=draw_bottom)


def plot_metric_triplet(raw_map, cvae_map, diff_map, title_prefix, cmap_main, vmin, vmax, diff_cmap, diff_lim, out_path):
    raw_map = np.ma.masked_invalid(raw_map)
    cvae_map = np.ma.masked_invalid(cvae_map)
    diff_map = np.ma.masked_invalid(diff_map)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(raw_map, origin="upper", cmap=cmap_main, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{title_prefix} - 原始S2S")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(cvae_map, origin="upper", cmap=cmap_main, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{title_prefix} - CVAE订正")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(diff_map, origin="upper", cmap=diff_cmap, vmin=-diff_lim, vmax=diff_lim)
    axes[2].set_title(f"{title_prefix} - 差值(订正-原始)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_metric_triplet_cartopy(
    raw_map,
    cvae_map,
    diff_map,
    title_prefix,
    cmap_main,
    vmin,
    vmax,
    diff_cmap,
    diff_lim,
    lat,
    lon,
    out_path,
):
    proj = ccrs.PlateCarree()
    extent = [float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))]
    lon2d, lat2d = np.meshgrid(lon, lat)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), subplot_kw={"projection": proj})
    for idx, ax in enumerate(axes):
        _add_cartopy_base(ax, extent, draw_left=(idx == 0), draw_bottom=True)

    raw_map = apply_china_mask(raw_map, lat, lon)
    cvae_map = apply_china_mask(cvae_map, lat, lon)
    diff_map = apply_china_mask(diff_map, lat, lon)

    im0 = axes[0].pcolormesh(
        lon2d, lat2d, raw_map, transform=proj, shading="auto", cmap=cmap_main, vmin=vmin, vmax=vmax, zorder=1
    )
    axes[0].set_title(f"{title_prefix} - 原始S2S", fontsize=10)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].pcolormesh(
        lon2d, lat2d, cvae_map, transform=proj, shading="auto", cmap=cmap_main, vmin=vmin, vmax=vmax, zorder=1
    )
    axes[1].set_title(f"{title_prefix} - CVAE订正", fontsize=10)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].pcolormesh(
        lon2d, lat2d, diff_map, transform=proj, shading="auto", cmap=diff_cmap, vmin=-diff_lim, vmax=diff_lim, zorder=1
    )
    axes[2].set_title(f"{title_prefix} - 差值(订正-原始)", fontsize=10)
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_metric_overview(records, metric_name, cmap_main, vmin, vmax, diff_cmap, diff_lim, out_path):
    """
    将多个lead的原始/CVAE/差值合并成一张总图。
    每一行对应一个lead，每一列分别是原始、订正和差值。
    """
    n = len(records)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.2 * n))
    if n == 1:
        axes = np.array([axes])

    for row, rec in enumerate(records):
        panels = [rec["raw"], rec["cvae"], rec["diff"]]
        labels = [chr(ord("a") + row * 3 + col) for col in range(3)]
        titles = [
            f"（{labels[0]}）Lead {rec['lead']}天 原始S2S",
            f"（{labels[1]}）Lead {rec['lead']}天 CVAE订正",
            f"（{labels[2]}）Lead {rec['lead']}天 差值(订正-原始)",
        ]
        for col in range(3):
            ax = axes[row, col]
            if col < 2:
                im = ax.imshow(np.ma.masked_invalid(panels[col]), origin="upper", cmap=cmap_main, vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(np.ma.masked_invalid(panels[col]), origin="upper", cmap=diff_cmap, vmin=-diff_lim, vmax=diff_lim)
            ax.set_title(titles[col], fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_metric_overview_cartopy(records, metric_name, cmap_main, vmin, vmax, diff_cmap, diff_lim, lat, lon, out_path):
    n = len(records)
    proj = ccrs.PlateCarree()
    extent = [float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))]
    lon2d, lat2d = np.meshgrid(lon, lat)
    fig, axes = plt.subplots(n, 3, figsize=(15.5, 4.4 * n), subplot_kw={"projection": proj})
    if n == 1:
        axes = np.array([axes])

    for row, rec in enumerate(records):
        panels = [rec["raw"], rec["cvae"], rec["diff"]]
        labels = [chr(ord("a") + row * 3 + col) for col in range(3)]
        titles = [
            f"（{labels[0]}）Lead {rec['lead']}天 原始S2S",
            f"（{labels[1]}）Lead {rec['lead']}天 CVAE订正",
            f"（{labels[2]}）Lead {rec['lead']}天 差值(订正-原始)",
        ]
        for col in range(3):
            ax = axes[row, col]
            _add_cartopy_base(ax, extent, draw_left=(col == 0), draw_bottom=(row == n - 1))
            if col < 2:
                data = apply_china_mask(panels[col], lat, lon)
                im = ax.pcolormesh(
                    lon2d, lat2d, data, transform=proj, shading="auto", cmap=cmap_main, vmin=vmin, vmax=vmax, zorder=1
                )
            else:
                data = apply_china_mask(panels[col], lat, lon)
                im = ax.pcolormesh(
                    lon2d,
                    lat2d,
                    data,
                    transform=proj,
                    shading="auto",
                    cmap=diff_cmap,
                    vmin=-diff_lim,
                    vmax=diff_lim,
                    zorder=1,
                )
            ax.set_title(titles[col], fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_rmse_overview(records, out_path):
    """
    RMSE总览图使用对数色标展示原始与订正结果，
    差值列改为 raw-cvae，使红色表示改进、蓝色表示退化。
    """
    n = len(records)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.2 * n))
    if n == 1:
        axes = np.array([axes])

    rmse_all = np.concatenate(
        [x["raw"].ravel() for x in records] +
        [x["cvae"].ravel() for x in records]
    )
    positive = rmse_all[rmse_all > 0]
    vmin = max(float(np.nanpercentile(positive, 1)), 1e-3)
    vmax = float(np.nanpercentile(positive, 99))
    main_norm = LogNorm(vmin=vmin, vmax=vmax)

    improve_all = np.concatenate([(x["raw"] - x["cvae"]).ravel() for x in records])
    diff_lim = max(float(np.nanpercentile(np.abs(improve_all), 99)), 0.05)

    for row, rec in enumerate(records):
        panels = [rec["raw"], rec["cvae"], rec["raw"] - rec["cvae"]]
        labels = [chr(ord("a") + row * 3 + col) for col in range(3)]
        titles = [
            f"（{labels[0]}）Lead {rec['lead']}天 原始S2S",
            f"（{labels[1]}）Lead {rec['lead']}天 CVAE订正",
            f"（{labels[2]}）Lead {rec['lead']}天 改善量(原始-订正)",
        ]
        for col in range(3):
            ax = axes[row, col]
            if col < 2:
                im = ax.imshow(np.ma.masked_invalid(panels[col]), origin="upper", cmap="viridis", norm=main_norm)
            else:
                im = ax.imshow(np.ma.masked_invalid(panels[col]), origin="upper", cmap="RdBu_r", vmin=-diff_lim, vmax=diff_lim)
            ax.set_title(titles[col], fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = plt.colorbar(im, ax=ax, fraction=0.046)
            if col < 2:
                cbar.set_label("RMSE (mm/day)")
            else:
                cbar.set_label("Improvement (mm/day)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_rmse_overview_cartopy(records, lat, lon, out_path):
    n = len(records)
    proj = ccrs.PlateCarree()
    extent = [float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))]
    lon2d, lat2d = np.meshgrid(lon, lat)
    fig, axes = plt.subplots(n, 3, figsize=(15.5, 4.4 * n), subplot_kw={"projection": proj})
    if n == 1:
        axes = np.array([axes])

    rmse_all = np.concatenate([x["raw"].ravel() for x in records] + [x["cvae"].ravel() for x in records])
    positive = rmse_all[rmse_all > 0]
    vmin = max(float(np.nanpercentile(positive, 1)), 1e-3)
    vmax = float(np.nanpercentile(positive, 99))
    main_norm = LogNorm(vmin=vmin, vmax=vmax)

    improve_all = np.concatenate([(x["raw"] - x["cvae"]).ravel() for x in records])
    diff_lim = max(float(np.nanpercentile(np.abs(improve_all), 99)), 0.05)

    for row, rec in enumerate(records):
        panels = [rec["raw"], rec["cvae"], rec["raw"] - rec["cvae"]]
        labels = [chr(ord("a") + row * 3 + col) for col in range(3)]
        titles = [
            f"（{labels[0]}）Lead {rec['lead']}天 原始S2S",
            f"（{labels[1]}）Lead {rec['lead']}天 CVAE订正",
            f"（{labels[2]}）Lead {rec['lead']}天 改善量(原始-订正)",
        ]
        for col in range(3):
            ax = axes[row, col]
            _add_cartopy_base(ax, extent, draw_left=(col == 0), draw_bottom=(row == n - 1))
            if col < 2:
                data = apply_china_mask(panels[col], lat, lon)
                im = ax.pcolormesh(
                    lon2d, lat2d, data, transform=proj, shading="auto", cmap="viridis", norm=main_norm, zorder=1
                )
            else:
                data = apply_china_mask(panels[col], lat, lon)
                im = ax.pcolormesh(
                    lon2d,
                    lat2d,
                    data,
                    transform=proj,
                    shading="auto",
                    cmap="RdBu_r",
                    vmin=-diff_lim,
                    vmax=diff_lim,
                    zorder=1,
                )
            ax.set_title(titles[col], fontsize=10)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046)
            if col < 2:
                cbar.set_label("RMSE (mm/day)")
            else:
                cbar.set_label("Improvement (mm/day)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def write_summary_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_leadtime(
    data_dir="data/processed",
    s2s_path="data/processed/s2s_china_summer.nc",
    era5_path="data/processed/era5_china_summer.nc",
    output_dir="figures/leadtime_skill",
    lead_days=(1, 5, 10, 15, 20),
    n_ens=10,
):
    setup_chinese_font()
    os.makedirs(output_dir, exist_ok=True)

    model, device, x_mean, x_std, y_mean, y_std = load_model_and_norm(data_dir)
    lead_pairs, lat, lon, available_leads = collect_pairs_by_lead(s2s_path, era5_path, lead_days)

    summary_rows = []
    missing = []
    acc_overview_records = []
    rmse_overview_records = []

    for lead in lead_days:
        item = lead_pairs[lead]
        if item["X"] is None or item["n"] == 0:
            missing.append(lead)
            continue

        X = item["X"]
        Y = item["Y"]
        Y_pred = predict_cvae(model, device, X, x_mean, x_std, y_mean, y_std, n_ens=n_ens)

        rmse_raw = rmse_map(X, Y)
        rmse_cvae = rmse_map(Y_pred, Y)
        rmse_diff = rmse_cvae - rmse_raw
        rmse_improve_pct = (rmse_raw - rmse_cvae) / (rmse_raw + 1e-8) * 100.0

        acc_raw = acc_map(X, Y)
        acc_cvae = acc_map(Y_pred, Y)
        acc_diff = acc_cvae - acc_raw

        acc_raw_masked = apply_china_mask(acc_raw, lat, lon)
        acc_cvae_masked = apply_china_mask(acc_cvae, lat, lon)
        acc_diff_masked = apply_china_mask(acc_diff, lat, lon)
        rmse_raw_masked = apply_china_mask(rmse_raw, lat, lon)
        rmse_cvae_masked = apply_china_mask(rmse_cvae, lat, lon)
        rmse_diff_masked = apply_china_mask(rmse_diff, lat, lon)
        rmse_improve_pct_masked = apply_china_mask(rmse_improve_pct, lat, lon)

        # 图：ACC（[-1,1]）和RMSE（主图用统一范围）
        rmse_vmax = float(np.nanpercentile(np.concatenate([rmse_raw.ravel(), rmse_cvae.ravel()]), 99))
        rmse_vmax = max(rmse_vmax, 0.1)
        rmse_diff_lim = float(np.nanpercentile(np.abs(rmse_diff), 99))
        rmse_diff_lim = max(rmse_diff_lim, 0.05)

        acc_diff_lim = float(np.nanpercentile(np.abs(acc_diff), 99))
        acc_diff_lim = max(acc_diff_lim, 0.02)

        acc_path = os.path.join(output_dir, f"lead{lead:02d}_acc_compare.png")
        rmse_path = os.path.join(output_dir, f"lead{lead:02d}_rmse_compare.png")
        acc_cartopy_path = os.path.join(output_dir, f"lead{lead:02d}_acc_compare_cartopy.png")
        rmse_cartopy_path = os.path.join(output_dir, f"lead{lead:02d}_rmse_compare_cartopy.png")
        rmse_imp_path = os.path.join(output_dir, f"lead{lead:02d}_rmse_improve_pct.png")
        npz_path = os.path.join(output_dir, f"lead{lead:02d}_skill_maps.npz")

        plot_metric_triplet(
            raw_map=acc_raw_masked,
            cvae_map=acc_cvae_masked,
            diff_map=acc_diff_masked,
            title_prefix=f"Lead {lead}天 ACC",
            cmap_main="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            diff_cmap="RdBu_r",
            diff_lim=acc_diff_lim,
            out_path=acc_path,
        )
        plot_metric_triplet_cartopy(
            raw_map=acc_raw_masked,
            cvae_map=acc_cvae_masked,
            diff_map=acc_diff_masked,
            title_prefix=f"Lead {lead}天 ACC",
            cmap_main="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            diff_cmap="RdBu_r",
            diff_lim=acc_diff_lim,
            lat=lat,
            lon=lon,
            out_path=acc_cartopy_path,
        )
        plot_metric_triplet(
            raw_map=rmse_raw_masked,
            cvae_map=rmse_cvae_masked,
            diff_map=rmse_diff_masked,
            title_prefix=f"Lead {lead}天 RMSE (mm/day)",
            cmap_main="viridis",
            vmin=0.0,
            vmax=rmse_vmax,
            diff_cmap="RdBu_r",
            diff_lim=rmse_diff_lim,
            out_path=rmse_path,
        )
        plot_metric_triplet_cartopy(
            raw_map=rmse_raw_masked,
            cvae_map=rmse_cvae_masked,
            diff_map=rmse_diff_masked,
            title_prefix=f"Lead {lead}天 RMSE (mm/day)",
            cmap_main="viridis",
            vmin=0.0,
            vmax=rmse_vmax,
            diff_cmap="RdBu_r",
            diff_lim=rmse_diff_lim,
            lat=lat,
            lon=lon,
            out_path=rmse_cartopy_path,
        )

        plt.figure(figsize=(6.2, 4.8))
        imp_lim = float(np.nanpercentile(np.abs(rmse_improve_pct), 99))
        imp_lim = max(imp_lim, 5.0)
        im = plt.imshow(np.ma.masked_invalid(rmse_improve_pct_masked), origin="upper", cmap="RdBu", vmin=-imp_lim, vmax=imp_lim)
        plt.title(f"Lead {lead}天 RMSE改善率(%)")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(im, fraction=0.046, label="%")
        plt.tight_layout()
        plt.savefig(rmse_imp_path, dpi=180, bbox_inches="tight")
        plt.close()

        np.savez_compressed(
            npz_path,
            rmse_raw=rmse_raw,
            rmse_cvae=rmse_cvae,
            rmse_improve_pct=rmse_improve_pct,
            acc_raw=acc_raw,
            acc_cvae=acc_cvae,
            acc_diff=acc_diff,
        )

        summary_rows.append(
            {
                "lead_day": lead,
                "n_samples": int(item["n"]),
                "spatial_mean_rmse_raw": float(np.nanmean(rmse_raw)),
                "spatial_mean_rmse_cvae": float(np.nanmean(rmse_cvae)),
                "spatial_mean_rmse_improve_%": float(np.nanmean(rmse_improve_pct)),
                "spatial_mean_acc_raw": float(np.nanmean(acc_raw)),
                "spatial_mean_acc_cvae": float(np.nanmean(acc_cvae)),
                "spatial_mean_acc_diff": float(np.nanmean(acc_diff)),
            }
        )

        acc_overview_records.append(
            {
                "lead": lead,
                "raw": acc_raw_masked,
                "cvae": acc_cvae_masked,
                "diff": acc_diff_masked,
            }
        )
        rmse_overview_records.append(
            {
                "lead": lead,
                "raw": rmse_raw_masked,
                "cvae": rmse_cvae_masked,
                "diff": rmse_diff_masked,
            }
        )

        print(f"[Lead {lead}] 样本数={item['n']} | ACC均值 {np.nanmean(acc_raw):.3f}->{np.nanmean(acc_cvae):.3f} | RMSE均值 {np.nanmean(rmse_raw):.3f}->{np.nanmean(rmse_cvae):.3f}")

    summary_csv = os.path.join(output_dir, "leadtime_skill_summary.csv")
    write_summary_csv(summary_csv, summary_rows)

    if acc_overview_records:
        acc_diff_lim = max(
            float(np.nanpercentile(np.abs(np.concatenate([x["diff"].ravel() for x in acc_overview_records])), 99)),
            0.02,
        )
        plot_metric_overview(
            records=acc_overview_records,
            metric_name="ACC",
            cmap_main="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            diff_cmap="RdBu_r",
            diff_lim=acc_diff_lim,
            out_path=os.path.join(output_dir, "leadtime_acc_overview.png"),
        )
        plot_metric_overview_cartopy(
            records=acc_overview_records,
            metric_name="ACC",
            cmap_main="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            diff_cmap="RdBu_r",
            diff_lim=acc_diff_lim,
            lat=lat,
            lon=lon,
            out_path=os.path.join(output_dir, "leadtime_acc_overview_cartopy.png"),
        )

    if rmse_overview_records:
        plot_rmse_overview(
            records=rmse_overview_records,
            out_path=os.path.join(output_dir, "leadtime_rmse_overview.png"),
        )
        plot_rmse_overview_cartopy(
            records=rmse_overview_records,
            lat=lat,
            lon=lon,
            out_path=os.path.join(output_dir, "leadtime_rmse_overview_cartopy.png"),
        )

    summary_txt = os.path.join(output_dir, "leadtime_skill_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Leadtime技巧评估汇总\n")
        f.write("=" * 36 + "\n")
        f.write(f"请求lead days: {list(lead_days)}\n")
        f.write(f"S2S可用lead days: {list(map(int, available_leads))}\n")
        if missing:
            f.write(f"无可用样本的lead days: {missing}\n")
            f.write("说明：当前S2S下载step仅覆盖1-14天，15/20天需扩展下载step后重跑流程。\n")
        if summary_rows:
            best_rmse = max(summary_rows, key=lambda x: x["spatial_mean_rmse_improve_%"])
            best_acc = max(summary_rows, key=lambda x: x["spatial_mean_acc_diff"])
            f.write(
                f"RMSE改善最明显: lead={best_rmse['lead_day']}天 "
                f"({best_rmse['spatial_mean_rmse_improve_%']:.2f}%)\n"
            )
            f.write(
                f"ACC提升最明显: lead={best_acc['lead_day']}天 "
                f"(+{best_acc['spatial_mean_acc_diff']:.3f})\n"
            )
        f.write("详细数据见 leadtime_skill_summary.csv 与各lead图件/npz。\n")

    print(f"已保存: {summary_csv}")
    print(f"已保存: {summary_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按lead day评估ACC/RMSE空间技巧")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--s2s_path", default="data/processed/s2s_china_summer.nc")
    parser.add_argument("--era5_path", default="data/processed/era5_china_summer.nc")
    parser.add_argument("--output_dir", default="figures/leadtime_skill")
    parser.add_argument("--lead_days", default="1,5,10,15,20", help="逗号分隔，如 1,5,10,15,20")
    parser.add_argument("--n_ens", type=int, default=10, help="CVAE采样次数")
    args = parser.parse_args()

    evaluate_leadtime(
        data_dir=args.data_dir,
        s2s_path=args.s2s_path,
        era5_path=args.era5_path,
        output_dir=args.output_dir,
        lead_days=tuple(parse_lead_days(args.lead_days)),
        n_ens=args.n_ens,
    )
