"""
ERA5降水数据下载脚本（作为观测数据）
需要注册Copernicus Climate Data Store账号
https://cds.climate.copernicus.eu/
"""

import cdsapi
import os

def download_era5_precipitation(
    year=2015,
    months=[6, 7, 8],
    area=[55, 70, 15, 140],  # North, West, South, East (中国区域)
    output_dir="data/era5_obs"
):
    """
    下载ERA5降水数据

    参数:
        year: 年份
        months: 月份列表
        area: 区域范围 [N, W, S, E]
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    c = cdsapi.Client()

    output_file = f"{output_dir}/era5_precip_{year}_summer.nc"

    print(f"开始下载ERA5数据: {year}年夏季")

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': 'total_precipitation',
            'year': str(year),
            'month': [f'{m:02d}' for m in months],
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': [f'{h:02d}:00' for h in range(0, 24)],
            'area': area,
            'format': 'netcdf',
        },
        output_file
    )

    print(f"下载完成: {output_file}")

if __name__ == "__main__":
    # 下载2015年夏季数据
    download_era5_precipitation(year=2015, months=[6, 7, 8])
