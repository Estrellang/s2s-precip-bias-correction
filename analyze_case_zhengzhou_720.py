"""
通用个例分析脚本
对命中目标日期的不同leadtime样本，比较S2S原始、CVAE订正与ERA5观测。
"""

import argparse
import csv
import os

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


def parse_int_list(text):
    if text is None:
        return None
    values = []
    for x in text.split(","):
        x = x.strip()
        if x:
            values.append(int(x))
    return values or None


def find_era5_on_date(era5, target_date):
    t = [x for x in era5.time.values if str(x)[:10] == target_date]
    if not t:
        return None
    return era5["tp"].sel(time=t[0]).values.astype(np.float32)


def collect_case_samples(s2s, target_date):
    samples = []
    for i, t in enumerate(s2s.time.values):
        for j, step in enumerate(s2s.step.values):
            valid_time = t + np.timedelta64(int(step), "ns")
            if str(valid_time)[:10] != target_date:
                continue
            lead_day = int(step / 1e9 / 3600 / 24)
            x = s2s["tp"].isel(time=i, step=j).values.astype(np.float32)
            x = np.nan_to_num(x, nan=0.0)
            samples.append(
                {
                    "init_date": str(t)[:10],
                    "lead_day": lead_day,
                    "x": x,
                }
            )
    samples = sorted(samples, key=lambda z: z["lead_day"])
    return samples


def load_model_norm(data_dir):
    norm = np.load(f"{data_dir}/norm_params.npy")
    x_mean, x_std, y_mean, y_std = norm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(latent_dim=64).to(device)
    model.load_state_dict(torch.load(f"{data_dir}/best_model.pt", map_location=device))
    model.eval()
    return model, device, x_mean, x_std, y_mean, y_std


def predict_cvae(model, device, x, x_mean, x_std, y_mean, y_std, n_ens=30):
    x_norm = (x - x_mean) / x_std
    x_tensor = torch.tensor(x_norm[None, None]).to(device)
    preds = []
    with torch.no_grad():
        for _ in range(n_ens):
            pred = model.predict(x_tensor).squeeze(1).squeeze(0).cpu().numpy()
            pred = pred * y_std + y_mean
            preds.append(pred)
    return np.mean(preds, axis=0).astype(np.float32)


def _add_cartopy_base(ax, extent, draw_left=True, draw_bottom=True):
    add_china_map_base(ax, extent, draw_left=draw_left, draw_bottom=draw_bottom)


def save_case_maps(cases, obs, output_png, point_idx, lat, lon, case_title, lat_window=6.0, lon_window=8.0):
    n = len(cases)
    pi, pj = point_idx

    lat_mask = (lat >= lat[pi] - lat_window) & (lat <= lat[pi] + lat_window)
    lon_mask = (lon >= lon[pj] - lon_window) & (lon <= lon[pj] + lon_window)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    def crop(field):
        return field[np.ix_(lat_idx, lon_idx)]

    cropped_lat = lat[lat_idx]
    cropped_lon = lon[lon_idx]
    cropped_obs = apply_china_mask(crop(obs), cropped_lat, cropped_lon)
    cropped_raw = [apply_china_mask(crop(c["raw"]), cropped_lat, cropped_lon) for c in cases]
    cropped_cvae = [apply_china_mask(crop(c["cvae"]), cropped_lat, cropped_lon) for c in cases]

    all_vals = np.concatenate(
        [cropped_obs.ravel()] +
        [x.ravel() for x in cropped_raw] +
        [x.ravel() for x in cropped_cvae]
    )
    positive = all_vals[all_vals > 0]
    vmin = max(float(np.nanpercentile(positive, 5)), 0.05)
    vmax = max(float(np.nanpercentile(positive, 99)), 1.0)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(n, 3, figsize=(12, 3.2 * n))
    if n == 1:
        axes = np.array([axes])

    local_i = int(np.where(lat_idx == pi)[0][0])
    local_j = int(np.where(lon_idx == pj)[0][0])
    for r, c in enumerate(cases):
        maps = [cropped_raw[r], cropped_cvae[r], cropped_obs]
        titles = [
            f"Lead {c['lead_day']}d 原始S2S\n(init {c['init_date']})",
            f"Lead {c['lead_day']}d CVAE订正",
            "ERA5观测",
        ]
        for k in range(3):
            ax = axes[r, k]
            im = ax.imshow(maps[k], origin="upper", cmap="Blues", norm=norm)
            ax.scatter([local_j], [local_i], c="red", s=18, marker="x")
            ax.set_title(titles[k], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 2:
                plt.colorbar(im, ax=ax, fraction=0.046, label="mm/day (log scale)")

    plt.suptitle(case_title, y=0.995, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close()


def save_case_maps_cartopy(cases, obs, output_png, point_idx, lat, lon, case_title, lat_window=6.0, lon_window=8.0):
    n = len(cases)
    pi, pj = point_idx

    lat_mask = (lat >= lat[pi] - lat_window) & (lat <= lat[pi] + lat_window)
    lon_mask = (lon >= lon[pj] - lon_window) & (lon <= lon[pj] + lon_window)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]

    def crop(field):
        return field[np.ix_(lat_idx, lon_idx)]

    cropped_lat = lat[lat_idx]
    cropped_lon = lon[lon_idx]
    cropped_obs = apply_china_mask(crop(obs), cropped_lat, cropped_lon)
    cropped_raw = [apply_china_mask(crop(c["raw"]), cropped_lat, cropped_lon) for c in cases]
    cropped_cvae = [apply_china_mask(crop(c["cvae"]), cropped_lat, cropped_lon) for c in cases]

    all_vals = np.concatenate([cropped_obs.ravel()] + [x.ravel() for x in cropped_raw] + [x.ravel() for x in cropped_cvae])
    positive = all_vals[all_vals > 0]
    vmax = max(float(np.nanpercentile(positive, 99)), 1.0)
    levels = np.linspace(0.0, vmax, 13)
    if levels[-1] <= levels[0]:
        levels = np.linspace(0.0, 1.0, 13)

    lon2d, lat2d = np.meshgrid(cropped_lon, cropped_lat)
    extent = [float(np.min(cropped_lon)), float(np.max(cropped_lon)), float(np.min(cropped_lat)), float(np.max(cropped_lat))]
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(n, 3, figsize=(12.8, 3.5 * n), subplot_kw={"projection": proj})
    if n == 1:
        axes = np.array([axes])

    point_lon = float(lon[pj])
    point_lat = float(lat[pi])

    for r, c in enumerate(cases):
        maps = [cropped_raw[r], cropped_cvae[r], cropped_obs]
        labels = [chr(ord("a") + r * 3 + k) for k in range(3)]
        titles = [
            f"（{labels[0]}）Lead {c['lead_day']}d 原始S2S\n(init {c['init_date']})",
            f"（{labels[1]}）Lead {c['lead_day']}d CVAE订正",
            f"（{labels[2]}）ERA5观测",
        ]
        for k in range(3):
            ax = axes[r, k]
            _add_cartopy_base(ax, extent, draw_left=(k == 0), draw_bottom=(r == n - 1))
            im = ax.contourf(
                lon2d,
                lat2d,
                maps[k],
                transform=proj,
                cmap="Blues",
                levels=levels,
                extend="max",
                zorder=1,
            )
            ax.scatter([point_lon], [point_lat], transform=proj, c="red", s=18, marker="x", zorder=4)
            ax.set_title(titles[k], fontsize=10)
            if k == 2:
                plt.colorbar(im, ax=ax, fraction=0.046, label="mm/day")

    plt.tight_layout()
    plt.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close()


def save_point_plot(cases, obs_point, output_png, city_name):
    leads = [c["lead_day"] for c in cases]
    raw_vals = [c["raw_point"] for c in cases]
    cvae_vals = [c["cvae_point"] for c in cases]
    raw_err = [abs(c["raw_point"] - obs_point) for c in cases]
    cvae_err = [abs(c["cvae_point"] - obs_point) for c in cases]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(leads, raw_vals, "o-", color="tab:red", label="S2S原始")
    ax1.plot(leads, cvae_vals, "o-", color="tab:blue", label="CVAE订正")
    ax1.axhline(obs_point, color="black", linestyle="--", label=f"ERA5观测={obs_point:.2f}")
    ax1.set_ylabel("降水 (mm/day)")
    ax1.set_title(f"{city_name}点位：不同leadtime预报值")
    ax1.legend()
    ax1.grid(alpha=0.25)
    for x, y in zip(leads, raw_vals):
        ax1.text(x, y + 1.0, f"{y:.1f}", color="tab:red", fontsize=9, ha="center")
    for x, y in zip(leads, cvae_vals):
        ax1.text(x, y + 0.35, f"{y:.1f}", color="tab:blue", fontsize=9, ha="center")

    width = 0.36
    x = np.arange(len(leads))
    ax2.bar(x - width / 2, raw_err, width=width, color="tab:red", alpha=0.6, label="S2S绝对误差")
    ax2.bar(x + width / 2, cvae_err, width=width, color="tab:blue", alpha=0.6, label="CVAE绝对误差")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(ld) for ld in leads])
    ax2.set_xlabel("Lead day")
    ax2.set_ylabel("|误差| (mm/day)")
    ax2.set_title(f"{city_name}点位：绝对误差对比")
    ax2.legend()
    ax2.grid(alpha=0.25)
    for idx, y in enumerate(raw_err):
        ax2.text(idx - width / 2, y + 0.6, f"{y:.1f}", fontsize=9, color="tab:red", ha="center")
    for idx, y in enumerate(cvae_err):
        ax2.text(idx + width / 2, y + 0.6, f"{y:.1f}", fontsize=9, color="tab:blue", ha="center")

    plt.tight_layout()
    plt.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close()


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_case_prefix(city_name, target_date):
    date_token = target_date.replace("-", "")
    city_token = city_name.lower()
    translit = {
        "郑州": "zhengzhou",
        "成都": "chengdu",
        "北京": "beijing",
        "广州": "guangzhou",
        "武汉": "wuhan",
        "长沙": "changsha",
    }
    city_token = translit.get(city_name, city_token)
    return f"{city_token}_{date_token}"


def analyze_case(
    target_date="2021-07-20",
    data_dir="data/processed",
    s2s_path="data/processed/s2s_china_summer.nc",
    era5_path="data/processed/era5_china_summer.nc",
    output_dir="figures/case_zhengzhou_20210720",
    city_name="郑州",
    city_lat=34.75,
    city_lon=113.62,
    n_ens=30,
    display_leads=None,
):
    setup_chinese_font()
    os.makedirs(output_dir, exist_ok=True)

    s2s = xr.open_dataset(s2s_path, decode_timedelta=False)
    era5 = xr.open_dataset(era5_path)

    obs = find_era5_on_date(era5, target_date)
    if obs is None:
        raise RuntimeError(f"ERA5中找不到目标日期: {target_date}")

    lat = era5["lat"].values
    lon = era5["lon"].values
    i = int(np.argmin(np.abs(lat - city_lat)))
    j = int(np.argmin(np.abs(lon - city_lon)))

    samples = collect_case_samples(s2s, target_date)
    if not samples:
        raise RuntimeError(f"S2S中找不到valid date={target_date}的样本")

    model, device, x_mean, x_std, y_mean, y_std = load_model_norm(data_dir)

    obs_point = float(obs[i, j])
    rows = []
    cases = []
    for item in samples:
        raw = item["x"]
        cvae = predict_cvae(model, device, raw, x_mean, x_std, y_mean, y_std, n_ens=n_ens)

        raw_point = float(raw[i, j])
        cvae_point = float(cvae[i, j])
        raw_abs_err = abs(raw_point - obs_point)
        cvae_abs_err = abs(cvae_point - obs_point)
        raw_rmse = float(np.sqrt(np.mean((raw - obs) ** 2)))
        cvae_rmse = float(np.sqrt(np.mean((cvae - obs) ** 2)))

        rows.append(
            {
                "target_date": target_date,
                "init_date": item["init_date"],
                "lead_day": item["lead_day"],
                "city": city_name,
                "grid_lat": float(lat[i]),
                "grid_lon": float(lon[j]),
                "obs_point_mmday": obs_point,
                "raw_point_mmday": raw_point,
                "cvae_point_mmday": cvae_point,
                "raw_abs_err": raw_abs_err,
                "cvae_abs_err": cvae_abs_err,
                "raw_domain_rmse": raw_rmse,
                "cvae_domain_rmse": cvae_rmse,
                "domain_rmse_improve_%": (raw_rmse - cvae_rmse) / (raw_rmse + 1e-8) * 100.0,
            }
        )
        cases.append(
            {
                "init_date": item["init_date"],
                "lead_day": item["lead_day"],
                "raw": raw,
                "cvae": cvae,
                "raw_point": raw_point,
                "cvae_point": cvae_point,
            }
        )

        print(
            f"lead={item['lead_day']:>2} init={item['init_date']} | "
            f"{city_name}点 obs={obs_point:.3f} raw={raw_point:.3f} cvae={cvae_point:.3f} | "
            f"abs_err {raw_abs_err:.3f}->{cvae_abs_err:.3f}"
        )

    case_prefix = build_case_prefix(city_name, target_date)
    csv_path = os.path.join(output_dir, f"{case_prefix}_case_stats.csv")
    maps_path = os.path.join(output_dir, f"{case_prefix}_maps_by_lead.png")
    maps_cartopy_path = os.path.join(output_dir, f"{case_prefix}_maps_by_lead_cartopy.png")
    point_path = os.path.join(output_dir, f"{case_prefix}_point_lead_compare.png")
    txt_path = os.path.join(output_dir, f"{case_prefix}_case_summary.txt")

    cases_for_maps = cases
    if display_leads:
        display_leads = list(dict.fromkeys(display_leads))
        cases_for_maps = [c for c in cases if c["lead_day"] in display_leads]
        if not cases_for_maps:
            raise RuntimeError(f"指定的display_leads={display_leads}没有命中任何可用个例样本")

    write_csv(csv_path, rows)
    save_case_maps(
        cases_for_maps,
        obs,
        maps_path,
        point_idx=(i, j),
        lat=lat,
        lon=lon,
        case_title=f"{target_date}{city_name}附近区域降水个例对比",
    )
    save_case_maps_cartopy(
        cases_for_maps,
        obs,
        maps_cartopy_path,
        point_idx=(i, j),
        lat=lat,
        lon=lon,
        case_title=f"{target_date}{city_name}附近区域降水个例对比",
    )
    save_point_plot(cases, obs_point, point_path, city_name=city_name)

    best_point = min(rows, key=lambda x: x["cvae_abs_err"])
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{target_date} {city_name}强降水个例分析\n")
        f.write("=" * 34 + "\n")
        f.write(f"目标日期: {target_date}\n")
        f.write(f"点位: {city_name} ({city_lat:.3f}N, {city_lon:.3f}E)\n")
        f.write(f"最近网格: ({lat[i]:.2f}N, {lon[j]:.2f}E)\n")
        f.write(f"ERA5该网格日降水: {obs_point:.3f} mm/day\n")
        f.write(f"可用lead: {[r['lead_day'] for r in rows]}\n")
        f.write(
            f"点位最优lead(按订正后绝对误差): {best_point['lead_day']}天 "
            f"(init={best_point['init_date']}, |err|={best_point['cvae_abs_err']:.3f})\n"
        )
        f.write(f"注：该分析基于ERA5日尺度、约1.25°网格，无法直接反映{city_name}小时级极端强降水峰值。\n")

    print(f"已保存: {csv_path}")
    print(f"已保存: {maps_path}")
    print(f"已保存: {maps_cartopy_path}")
    print(f"已保存: {point_path}")
    print(f"已保存: {txt_path}")

    s2s.close()
    era5.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="强降水个例分析")
    parser.add_argument("--target_date", default="2021-07-20")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--s2s_path", default="data/processed/s2s_china_summer.nc")
    parser.add_argument("--era5_path", default="data/processed/era5_china_summer.nc")
    parser.add_argument("--output_dir", default="figures/case_zhengzhou_20210720")
    parser.add_argument("--city_name", default="郑州")
    parser.add_argument("--city_lat", type=float, default=34.75)
    parser.add_argument("--city_lon", type=float, default=113.62)
    parser.add_argument("--n_ens", type=int, default=30)
    parser.add_argument("--display_leads", default=None, help="仅用于空间图展示的lead列表，如 4,14")
    args = parser.parse_args()

    analyze_case(
        target_date=args.target_date,
        data_dir=args.data_dir,
        s2s_path=args.s2s_path,
        era5_path=args.era5_path,
        output_dir=args.output_dir,
        city_name=args.city_name,
        city_lat=args.city_lat,
        city_lon=args.city_lon,
        n_ens=args.n_ens,
        display_leads=parse_int_list(args.display_leads),
    )
