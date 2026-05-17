"""
数据对齐：将S2S预报与对应的ERA5观测配对
"""

import xarray as xr
import numpy as np

def align_data(s2s_path, era5_path, output_dir):
    """
    将S2S预报与ERA5观测在时间上对齐
    输出：X (S2S预报), Y (ERA5观测) 配对数组
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    s2s = xr.open_dataset(s2s_path, decode_timedelta=False)
    era5 = xr.open_dataset(era5_path)

    # 将ERA5时间转为日期索引（忽略小时）
    era5_dates = {str(t)[:10]: t for t in era5.time.values}

    X_list, Y_list = [], []

    for i, t in enumerate(s2s.time.values):
        for j, step in enumerate(s2s.step.values):
            valid_time = t + np.timedelta64(int(step), 'ns')
            valid_date = str(valid_time)[:10]  # 只取日期部分

            if valid_date in era5_dates:
                x = s2s['tp'].isel(time=i, step=j).values
                y = era5['tp'].sel(time=era5_dates[valid_date]).values
                x = np.nan_to_num(x, nan=0.0)  # 边缘NaN填0
                X_list.append(x)
                Y_list.append(y)

    X = np.array(X_list)  # (N, lat, lon)
    Y = np.array(Y_list)  # (N, lat, lon)

    print(f"配对样本数: {len(X)}")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    np.save(f"{output_dir}/X_s2s.npy", X)
    np.save(f"{output_dir}/Y_era5.npy", Y)

    return X, Y

if __name__ == "__main__":
    align_data(
        s2s_path="data/processed/s2s_china_summer.nc",
        era5_path="data/processed/era5_china_summer.nc",
        output_dir="data/processed"
    )
