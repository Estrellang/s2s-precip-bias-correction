# S2S Precipitation Bias Correction with CVAE

基于条件变分自编码器（Conditional Variational Autoencoder, CVAE）的 S2S 降水预报偏差订正项目。该项目使用 ECMWF S2S 降水预报作为输入，以 ERA5 再分析降水作为观测真值，面向中国区域夏季（6-8 月）降水开展偏差订正、评估与可视化分析。

## 项目目标

- 对 ECMWF S2S 降水预报进行统计后处理与偏差订正
- 学习 S2S 预报与 ERA5 观测之间的非线性映射关系
- 评估订正后结果在 RMSE、相关系数和系统偏差上的改善
- 为中国区域夏季降水的次季节预报研究提供可复现流程

## 方法概览

项目核心模型为 CVAE：

- 编码器输入：`S2S 预报 + ERA5 观测`
- 潜变量：学习预报误差和观测分布特征
- 解码器输入：`潜变量 + S2S 预报`
- 输出：订正后的降水场

训练目标由两部分组成：

- 重构损失：MSE
- KL 散度正则项：`0.001 × KL`

推理阶段会从标准正态分布多次采样潜变量，并对多次预测结果取均值，以降低随机性。

## 研究区域与数据

- 区域范围：中国区域 `15-55°N, 70-140°E`
- 季节范围：夏季 `6-8 月`
- S2S 数据：ECMWF S2S 降水预报
- 观测真值：ERA5 再分析降水
- 统一网格：插值到 ERA5 网格，约 `33 × 57`

## 工作流

完整流程如下：

```text
API 配置
  ↓
数据下载
  ↓
数据预处理
  ↓
时间对齐
  ↓
模型训练
  ↓
结果评估
  ↓
可视化分析
```

对应脚本如下：

1. `setup_ecmwf_key.py`：配置 ECMWF API key
2. `setup_cds_key.py`：配置 CDS API key
3. `download_s2s.py`：下载 S2S 数据
4. `download_era5.py`：下载 ERA5 数据
5. `batch_download.py`：批量下载多年份数据
6. `preprocess.py`：区域裁剪、季节筛选、插值与单位处理
7. `align_data.py`：将 S2S 预报与 ERA5 观测按有效时间配对
8. `train.py`：训练 CVAE 模型
9. `evaluate.py`：评估订正效果
10. `visualize.py`：生成空间图、时间序列图、误差分布图等

## 目录结构

```text
.
├── align_data.py
├── batch_download.py
├── china_map.py
├── download_era5.py
├── download_s2s.py
├── evaluate.py
├── model.py
├── preprocess.py
├── requirements.txt
├── setup_ecmwf_key.py
├── train.py
├── visualize.py
├── data/
│   ├── s2s_raw/
│   ├── era5_obs/
│   └── processed/
└── figures/
```

说明：

- `data/s2s_raw/`：原始 S2S GRIB 数据
- `data/era5_obs/`：原始 ERA5 NetCDF 数据
- `data/processed/`：预处理结果、对齐样本、模型权重和标准化参数
- `figures/`：评估图件与案例分析结果

## 环境依赖

建议使用 Python 3.10 及以上版本。

安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：

- `torch`
- `xarray`
- `numpy`
- `scipy`
- `matplotlib`
- `cartopy`
- `cfgrib`
- `ecmwf-api-client`
- `cdsapi`

## 快速开始

### 1. 配置 API

首次使用前，需要分别配置 ECMWF 和 CDS 的 API key：

```bash
python setup_ecmwf_key.py
python setup_cds_key.py
```

如果你还没有账号：

- ECMWF: [https://apps.ecmwf.int/registration/](https://apps.ecmwf.int/registration/)
- CDS: [https://cds.climate.copernicus.eu/user/register](https://cds.climate.copernicus.eu/user/register)

### 2. 下载数据

单年份下载：

```bash
python download_s2s.py
python download_era5.py
```

批量下载：

```bash
python batch_download.py
```

### 3. 数据预处理

将原始 S2S 和 ERA5 数据统一到中国区域和相同网格：

```bash
python preprocess.py
```

该步骤主要完成：

- 裁剪中国区域
- 筛选夏季月份
- 将 S2S 插值到 ERA5 网格
- 将降水单位统一为 `mm/day`

### 4. 数据时间对齐

将 S2S 预报的有效时间与 ERA5 观测日期配对：

```bash
python align_data.py
```

输出文件：

- `data/processed/X_s2s.npy`
- `data/processed/Y_era5.npy`

### 5. 训练模型

```bash
python train.py
```

默认训练设置：

- `latent_dim = 64`
- `batch_size = 32`
- `learning_rate = 1e-3`
- 数据划分：训练/验证/测试 = `7:1:2`

训练完成后将保存：

- `data/processed/best_model.pt`
- `data/processed/norm_params.npy`

### 6. 模型评估

```bash
python evaluate.py
```

默认输出指标包括：

- 原始 S2S 的 RMSE
- CVAE 订正后的 RMSE
- 相关系数
- 偏差
- RMSE 改善百分比

### 7. 结果可视化

```bash
python visualize.py
```

会生成空间分布图、误差分布图、时间序列图等图件，输出到 `figures/` 目录。

## 模型说明

`model.py` 中定义了 CVAE 结构：

- `Encoder`：输入 `2` 通道场，提取潜变量分布参数 `mu` 和 `logvar`
- `Decoder`：输入潜变量和条件场，恢复订正后的降水
- `CVAE.predict()`：推理阶段从 `N(0,1)` 采样潜变量并生成结果

训练损失定义在 `train.py`：

```text
Loss = MSE(reconstruction, target) + 0.001 × KL
```

## 当前仓库说明

由于原始数据、处理后数组和模型权重体积较大，这些内容默认不会上传到 GitHub。也就是说，克隆本仓库后通常需要你自行重新下载并处理数据。

如果你希望让别人复现实验，建议至少提供：

- 数据获取说明
- 运行顺序
- 关键参数设置
- 评估指标说明

本仓库已经包含上述主要流程脚本，可直接按本 README 顺序执行。

## 注意事项

- S2S 和 ERA5 数据下载较慢，建议先用 1-2 年数据测试流程
- 每年数据可能需要约 `5-10 GB` 磁盘空间
- `preprocess.py` 中对 ERA5 降水做了单位换算，请确保输入数据类型一致
- `align_data.py` 通过有效时间匹配 ERA5 日期，时间坐标格式需要正确
- 推理时必须使用训练阶段保存的标准化参数 `norm_params.npy`

## 后续可改进方向

- 补充实验配置文件，减少脚本中的硬编码路径
- 增加固定随机种子和可复现实验设置
- 增加更完整的测试集划分与独立评估脚本
- 补充案例结果图和论文式说明图
- 增加英文 README 或中英文双语文档

## 引用与致谢

如果这个项目对你的研究有帮助，欢迎在使用时注明数据来源与方法来源：

- ECMWF S2S Forecast
- ERA5 Reanalysis
- Conditional Variational Autoencoder

