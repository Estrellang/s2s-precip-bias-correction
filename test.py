import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ================= 配置参数 =================
file_path = "data/ERA5.daily.L125.tp.1979-2023.nc"
target_date_str = "2021-07-20"
zhengzhou_lon = 113.6
zhengzhou_lat = 34.7

# ================= 1. 读取数据 =================
print(f"正在读取文件: {file_path} ...")
ds = xr.open_dataset(file_path)

# 检查变量名 (通常是 tp)
var_name = 'tp'
if var_name not in ds.variables:
    print(f"❌ 错误：变量 '{var_name}' 不在文件中。文件包含: {list(ds.variables)}")
    exit()

# ================= 2. 选取时间 (自动寻找最近的时间) =================
# 使用 method='nearest' 可以避免因时间标签微小差异导致的报错
try:
    # 先选取时间
    ds_time = ds.sel(time=target_date_str, method='nearest')

    # 打印实际选中的时间，确认是否偏差太大
    actual_time = ds_time.time.values
    print(f"🔍 目标时间: {target_date_str}")
    print(f"📅 实际选中时间: {np.datetime_as_string(actual_time, unit='D')}")

    # 选取郑州附近的点 (注意：这里使用 'lon' 和 'lat'，而不是 longitude/latitude)
    point_data = ds_time.sel(lon=zhengzhou_lon, lat=zhengzhou_lat, method='nearest')

    # 提取降水值 (单位通常是米，乘以1000变为毫米)
    precip_val = point_data[var_name].values * 1000

    print("-" * 30)
    print(f"📍 郑州最近格点坐标: ({point_data.lon.values:.2f}, {point_data.lat.values:.2f})")
    print(f"🌧️  该格点 2021-07-20 降水值: {precip_val:.2f} mm/day")
    print("-" * 30)

    # ================= 3. 绘图验证 =================
    # 提取当天的全场数据进行绘图
    data_to_plot = ds_time[var_name] * 1000  # 转为 mm

    plt.figure(figsize=(10, 6))
    # 使用 PlateCarree 投影
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 绘图 (限制在中国附近区域)
    # 注意：ERA5 L125 的经度可能是 0-360，也可能是 -180-180。
    # 如果报错或图是空的，可能需要调整 lon 的范围。这里假设是 0-360 或自动处理。
    mesh = data_to_plot.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="Blues",
        add_colorbar=True,
        cbar_kwargs={'label': 'Precipitation (mm/day)'},
        levels=20 # 颜色分级
    )

    # 添加海岸线和国界
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # 标出郑州位置
    ax.plot(zhengzhou_lon, zhengzhou_lat, 'r*', markersize=15, label='Zhengzhou', transform=ccrs.PlateCarree())
    ax.text(zhengzhou_lon + 1, zhengzhou_lat, 'Zhengzhou', color='red', fontsize=12, transform=ccrs.PlateCarree())

    plt.title(f"ERA5 Daily Precipitation: {np.datetime_as_string(actual_time, unit='D')}\n(注意格点数值)")
    plt.legend(loc='lower left')
    plt.show()

except Exception as e:
    print(f"❌ 发生错误: {e}")
    print("💡 提示：如果报错说坐标越界，可能是因为经度是 0-360 格式，而郑州是 113E。")