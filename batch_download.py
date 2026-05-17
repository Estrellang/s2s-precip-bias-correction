"""
批量下载多年夏季数据
"""

from download_s2s import download_s2s_precipitation

def batch_download(start_year=2015, end_year=2023):
    """
    批量下载多年夏季S2S数据
    （ERA5已有现成文件，无需下载）
    """
    for year in range(start_year, end_year + 1):
        print(f"\n{'='*50}")
        print(f"处理 {year} 年数据")
        print(f"{'='*50}\n")

        try:
            download_s2s_precipitation(
                start_date=f"{year}-06-01",
                end_date=f"{year}-08-31",
                output_dir="data/s2s_raw"
            )
        except Exception as e:
            print(f"S2S下载失败 {year}: {e}")

if __name__ == "__main__":
    batch_download(start_year=2015, end_year=2023)
