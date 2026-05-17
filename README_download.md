# 数据下载指南

## 前置准备

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 注册账号

**ECMWF账号（S2S数据）：**
- 访问：https://apps.ecmwf.int/registration/
- 注册后访问：https://api.ecmwf.int/v1/key/
- 获取API密钥

**CDS账号（ERA5数据）：**
- 访问：https://cds.climate.copernicus.eu/user/register
- 注册后访问：https://cds.climate.copernicus.eu/user
- 获取UID和API Key

### 3. 配置API密钥

```bash
# 配置ECMWF
python setup_ecmwf_key.py

# 配置CDS
python setup_cds_key.py
```

## 下载数据

### 方式1：单年下载

```bash
# 下载S2S数据
python download_s2s.py

# 下载ERA5数据
python download_era5.py
```

### 方式2：批量下载多年

```bash
python batch_download.py
```

## 数据说明

### S2S预报数据
- 路径：`data/s2s_raw/`
- 格式：GRIB
- 内容：ECMWF S2S降水预报

### ERA5观测数据
- 路径：`data/era5_obs/`
- 格式：NetCDF
- 内容：ERA5再分析降水（作为观测真值）

## 注意事项

1. 数据下载较慢，建议从少量数据开始测试
2. 首次下载建议只下载1-2年数据
3. 确保有足够的磁盘空间（每年约5-10GB）
4. 下载失败时检查API配置和网络连接
