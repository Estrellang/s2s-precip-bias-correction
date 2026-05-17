"""
S2S数据下载测试脚本 - 单日数据
"""

from ecmwfapi import ECMWFDataServer
import os

def test_download():
    """测试下载单日S2S数据"""

    os.makedirs("data/test", exist_ok=True)
    server = ECMWFDataServer()

    print("测试下载2020-06-01的S2S数据...")

    server.retrieve({
        "class": "s2",
        "dataset": "s2s",
        "date": "2020-06-01",
        "expver": "prod",
        "levtype": "sfc",
        "model": "glob",
        "origin": "ecmf",
        "param": "228228",
        "step": "0/24/48/72",
        "stream": "enfo",
        "time": "00:00:00",
        "type": "cf",
        "target": "data/test/s2s_test.grib",
    })

    print("测试完成！")

if __name__ == "__main__":
    test_download()
