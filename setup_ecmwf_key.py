"""
ECMWF API配置脚本
运行此脚本设置你的API密钥
"""

import os
from pathlib import Path

def setup_ecmwf_credentials():
    """
    设置ECMWF API凭证
    需要从 https://api.ecmwf.int/v1/key/ 获取你的密钥
    """
    print("请输入你的ECMWF API凭证：")
    print("(可以在 https://api.ecmwf.int/v1/key/ 找到)")

    url = input("API URL (默认: https://api.ecmwf.int/v1): ").strip()
    if not url:
        url = "https://api.ecmwf.int/v1"

    key = input("API Key: ").strip()
    email = input("Email: ").strip()

    # 创建配置文件
    ecmwfapirc_path = Path.home() / ".ecmwfapirc"

    with open(ecmwfapirc_path, 'w') as f:
        f.write('{\n')
        f.write(f'    "url"   : "{url}",\n')
        f.write(f'    "key"   : "{key}",\n')
        f.write(f'    "email" : "{email}"\n')
        f.write('}\n')

    print(f"\n配置文件已创建: {ecmwfapirc_path}")
    print("现在可以使用ECMWF API了！")

if __name__ == "__main__":
    setup_ecmwf_credentials()
