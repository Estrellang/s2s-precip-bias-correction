"""
数据预处理脚本
将S2S和ERA5数据处理为统一格式
"""

import xarray as xr
import numpy as np
import glob
import os

# 中国区域范围
LAT_MIN, LAT_MAX = 15, 55
LON_MIN, LON_MAX = 70, 140

def crop_china(ds, lat_name='latitude', lon_name='longitude'):
    """裁剪到中国区域"""
    return ds.sel(
        {lat_name: slice(LAT_MAX, LAT_MIN),
         lon_name: slice(LON_MIN, LON_MAX)}
    )

def process_era5(era5_path, output_dir):
    """处理ERA5数据：裁剪区域，筛选夏季"""
    os.makedirs(output_dir, exist_ok=True)

    print("处理ERA5数据...")
    ds = xr.open_dataset(era5_path)

    # 裁剪中国区域
    ds = crop_china(ds, lat_name='lat', lon_name='lon')

    # 筛选夏季（6-8月）
    ds = ds.sel(time=ds.time.dt.month.isin([6, 7, 8]))

    # 筛选2015-2023年
    ds = ds.sel(time=slice('2015-01-01', '2023-12-31'))

    # 当前ERA5日文件来自daymean，需先由m换算到mm，再乘24还原为日累计降水
    ds['tp'] = ds['tp'] * 1000 * 24
    ds['tp'].attrs['units'] = 'mm/day'

    output_path = f"{output_dir}/era5_china_summer.nc"
    ds.to_netcdf(output_path)
    print(f"ERA5处理完成: {output_path}")
    print(f"形状: {ds['tp'].shape}")
    return ds


def process_s2s(s2s_dir, era5_ref, output_dir):
    """处理S2S数据：裁剪区域，插值到ERA5网格"""
    os.makedirs(output_dir, exist_ok=True)

    grib_files = sorted(glob.glob(f"{s2s_dir}/*.grib"))
    datasets = []

    for f in grib_files:
        print(f"处理: {os.path.basename(f)}")
        ds = xr.open_dataset(f, engine='cfgrib')
        ds = crop_china(ds)
        # 取相邻step差值得到日降水，并去掉step=0
        tp_daily = ds['tp'].diff(dim='step').clip(min=0)
        ds = ds.isel(step=slice(1, None))  # 去掉step=0
        ds['tp'].values[:] = tp_daily.values
        ds['tp'].attrs['units'] = 'mm/day'
        ds = ds.interp(latitude=era5_ref.lat.values, longitude=era5_ref.lon.values)
        datasets.append(ds)

    combined = xr.concat(datasets, dim='time')
    output_path = f"{output_dir}/s2s_china_summer.nc"
    combined.to_netcdf(output_path)
    print(f"S2S处理完成: {output_path}, 形状: {combined['tp'].shape}")
    return combined


if __name__ == "__main__":
    era5 = process_era5(
        era5_path="data/ERA5.daily.L125.tp.1979-2023.nc",
        output_dir="data/processed"
    )
    s2s = process_s2s(
        s2s_dir="data/s2s_raw",
        era5_ref=era5,
        output_dir="data/processed"
    )
