"""
中国范围地图工具：
1. 读取中国国界几何
2. 生成中国范围掩膜
3. 为Cartopy地图添加仅中国边界的底图
"""

from functools import lru_cache

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep


CHINA_REGION_NAMES = {"China", "Taiwan"}


@lru_cache(maxsize=1)
def load_china_geometry():
    """
    从Natural Earth国家边界中读取中国几何对象。
    """
    shp_path = shpreader.natural_earth(
        resolution="50m",
        category="cultural",
        name="admin_0_countries",
    )

    reader = shpreader.Reader(shp_path)
    china_geoms = []
    for record in reader.records():
        attrs = record.attributes
        if (
            attrs.get("NAME_LONG") in CHINA_REGION_NAMES
            or attrs.get("ADMIN") in CHINA_REGION_NAMES
            or attrs.get("SOVEREIGNT") == "China"
        ):
            china_geoms.append(record.geometry)

    if not china_geoms:
        raise RuntimeError("未能从 Natural Earth 数据中找到中国边界。")

    return unary_union(china_geoms)


@lru_cache(maxsize=8)
def _cached_china_mask(lat_key, lon_key):
    lat = np.asarray(lat_key, dtype=float)
    lon = np.asarray(lon_key, dtype=float)
    lon2d, lat2d = np.meshgrid(lon, lat)

    china_geom = load_china_geometry()
    prepared = prep(china_geom)
    mask = np.zeros(lon2d.shape, dtype=bool)

    for i in range(lat2d.shape[0]):
        for j in range(lon2d.shape[1]):
            point = Point(float(lon2d[i, j]), float(lat2d[i, j]))
            mask[i, j] = prepared.contains(point) or china_geom.touches(point)

    return mask


def get_china_mask(lat, lon):
    """
    根据网格中心点判断是否位于中国范围内。
    """
    lat_key = tuple(np.round(np.asarray(lat, dtype=float), 6))
    lon_key = tuple(np.round(np.asarray(lon, dtype=float), 6))
    return _cached_china_mask(lat_key, lon_key).copy()


def apply_china_mask(field, lat, lon, fill_value=np.nan):
    """
    将中国范围外的格点设为 fill_value。
    """
    masked = np.array(field, dtype=float, copy=True)
    masked[~get_china_mask(lat, lon)] = fill_value
    return masked


def add_china_map_base(ax, extent, draw_left=True, draw_bottom=True):
    """
    只绘制中国边界，不再叠加周边国家国界，避免“中国图上出现其他国家着色”的问题。
    """
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    ax.add_geometries(
        [load_china_geometry()],
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="#2f2f2f",
        linewidth=0.8,
        zorder=3,
    )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.35,
        color="gray",
        alpha=0.45,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = draw_left
    gl.bottom_labels = draw_bottom
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
