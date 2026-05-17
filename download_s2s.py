"""
ECMWF S2S降水数据下载脚本
"""

from ecmwfapi import ECMWFDataServer
import os
from datetime import datetime, timedelta

def generate_s2s_dates(start_date, end_date):
    """生成S2S数据可用的日期（每周一和周四）"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = start
    while current <= end:
        # 0=周一, 3=周四
        if current.weekday() in [0, 3]:
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return "/".join(dates)

def download_s2s_precipitation(
    start_date="2015-06-01",
    end_date="2015-08-31",
    output_dir="data/s2s_raw"
):
    """
    下载ECMWF S2S降水预报数据

    参数:
        start_date: 起始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    server = ECMWFDataServer()

    output_file = f"{output_dir}/s2s_precip_{start_date}_{end_date}.grib"

    # 生成S2S可用日期
    date_string = generate_s2s_dates(start_date, end_date)

    print(f"开始下载数据: {start_date} 到 {end_date}")
    print(f"实际下载日期: {date_string}")

    server.retrieve({
        "class": "s2",
        "dataset": "s2s",
        "date": date_string,
        "expver": "prod",
        "levtype": "sfc",
        "model": "glob",
        "origin": "ecmf",
        "param": "228228",
        "step": "0/24/48/72/96/120/144/168/192/216/240/264/288/312/336",
        "stream": "enfo",
        "time": "00:00:00",
        "type": "cf",
        "target": output_file,
    })

    print(f"下载完成: {output_file}")

if __name__ == "__main__":
    # 示例：下载2015年夏季数据
    download_s2s_precipitation(
        start_date="2015-06-01",
        end_date="2015-08-31",
        output_dir="data/s2s_raw"
    )
