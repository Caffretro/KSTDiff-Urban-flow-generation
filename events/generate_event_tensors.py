"""
生成时空事件张量数据
将三类2019年事件（站点停运、演唱会、极端天气）转换为与模型输入对齐的时空事件张量

时间维度：按小时对齐，一年按 24×365 = 8760 小时索引
空间维度：MTR Voronoi 区域索引（103个区域）
事件通道：
  - service_outage: 区域是否受停运影响（0/1）
  - concert: 是否发生演唱会（0/1）
  - extreme_weather: 天气强度标量（0/1/2/3）
"""

import pandas as pd
import numpy as np
import json
import geopandas as gpd
from datetime import datetime, timedelta
from shapely.geometry import Point
from pathlib import Path

# 定义路径
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "data_HK"
EVENTS_DIR = DATA_DIR / "Events"
REGIONS_DIR = DATA_DIR / "HK_regions"

# 输出路径
OUTPUT_FILE = EVENTS_DIR / "hk_events_2019.npz"

# 2019年参数
YEAR = 2019
NUM_HOURS = 24 * 365  # 8760小时
NUM_REGIONS = 103  # MTR站点/Voronoi区域数量


def load_station_mapping():
    """加载MTR站点中英文名称和Station Code映射"""
    # 读取MTR网络数据
    mtr_df = pd.read_csv(REGIONS_DIR / "MTR_Service_Network.csv")

    # 创建中文名到Station Code的映射
    station_map = {}
    for _, row in mtr_df.iterrows():
        chinese_name = row['Chinese Name']
        station_code = row['Station Code']
        if chinese_name not in station_map:
            station_map[chinese_name] = station_code

    return station_map


def load_voronoi_regions():
    """加载Voronoi区域GeoDataFrame"""
    geojson_path = REGIONS_DIR / "mtr_voronoi_final.geojson"
    gdf = gpd.read_file(geojson_path)

    # 创建Station Code到区域索引的映射
    code_to_idx = {}
    for idx, row in gdf.iterrows():
        station_code = row['Station code']
        code_to_idx[station_code] = idx

    return gdf, code_to_idx


def datetime_to_hour_index(dt_str, year=2019):
    """
    将datetime字符串转换为年内小时索引

    Args:
        dt_str: datetime字符串，如 "2019-07-01 13:15"
        year: 年份

    Returns:
        hour_index: 0-8759之间的小时索引
    """
    dt = pd.to_datetime(dt_str)

    # 计算从年初到当前时间的小时数
    year_start = datetime(year, 1, 1)
    hours_since_start = int((dt - year_start).total_seconds() / 3600)

    return hours_since_start


def find_voronoi_for_location(lat, lon, gdf):
    """
    根据经纬度查找所属的Voronoi区域索引

    Args:
        lat: 纬度
        lon: 经度
        gdf: Voronoi区域GeoDataFrame

    Returns:
        区域索引，如果找不到返回None
    """
    point = Point(lon, lat)

    for idx, row in gdf.iterrows():
        if row['geometry'].contains(point):
            return idx

    # 如果点不在任何区域内，找最近的区域
    distances = gdf.geometry.distance(point)
    return distances.idxmin()


def process_service_outage(station_map, code_to_idx):
    """
    处理站点停运事件

    Returns:
        service_outage_tensor: (8760, 103) 二值张量
    """
    print("Processing service outage events...")

    tensor = np.zeros((NUM_HOURS, NUM_REGIONS), dtype=np.float32)

    # 读取停运数据
    outage_df = pd.read_csv(EVENTS_DIR / "mtr_events.txt")

    for _, row in outage_df.iterrows():
        station_name = row['station']
        start_time = row['start_time']
        end_time = row['end_time']

        # 获取Station Code
        if station_name not in station_map:
            print(f"Warning: Station '{station_name}' not found in mapping")
            continue

        station_code = station_map[station_name]

        # 获取区域索引
        if station_code not in code_to_idx:
            print(
                f"Warning: Station code '{station_code}' not found in Voronoi regions")
            continue

        region_idx = code_to_idx[station_code]

        # 计算时间索引
        start_hour = datetime_to_hour_index(start_time)
        end_hour = datetime_to_hour_index(end_time)

        # 确保索引在有效范围内
        start_hour = max(0, min(start_hour, NUM_HOURS - 1))
        end_hour = max(0, min(end_hour, NUM_HOURS - 1))

        # 填充张量（包含end_hour这一小时）
        tensor[start_hour:end_hour + 1, region_idx] = 1.0

    print(
        f"Service outage events: {np.sum(tensor > 0)} hour-region pairs affected")
    return tensor


def process_concerts(gdf):
    """
    处理演唱会事件

    Returns:
        concert_tensor: (8760, 103) 二值张量
    """
    print("Processing concert events...")

    tensor = np.zeros((NUM_HOURS, NUM_REGIONS), dtype=np.float32)

    # 读取演唱会数据
    concerts_df = pd.read_csv(
        EVENTS_DIR / "Hong_Kong_Concerts_2019_Final_processed.csv")

    for _, row in concerts_df.iterrows():
        try:
            # 解析日期时间
            show_datetime = row['Show_DateTime']
            lat = row['Latitude']
            lon = row['Longitude']

            # 处理时间格式（可能包含pm/am标记）
            # 示例: "2019-01-01 15:00pm" 或 "2019-01-01 3:00pm"
            dt_str = show_datetime.replace('pm', '').replace('am', '').strip()

            # 获取小时索引
            hour_idx = datetime_to_hour_index(dt_str)

            # 确保索引在有效范围内
            if hour_idx < 0 or hour_idx >= NUM_HOURS:
                continue

            # 查找所属Voronoi区域
            region_idx = find_voronoi_for_location(lat, lon, gdf)

            if region_idx is not None:
                # 标记该小时及之后的小时（直到当天结束）
                # 从演唱会开始时间到当天23:59
                dt = pd.to_datetime(dt_str)
                end_of_day = datetime(dt.year, dt.month, dt.day, 23, 59, 59)
                end_hour = datetime_to_hour_index(
                    end_of_day.strftime("%Y-%m-%d %H:%M"))

                # 确保end_hour在有效范围内
                end_hour = min(end_hour, NUM_HOURS - 1)

                tensor[hour_idx:end_hour + 1, region_idx] = 1.0

        except Exception as e:
            print(f"Error processing concert: {e}, row: {row}")
            continue

    print(f"Concert events: {np.sum(tensor > 0)} hour-region pairs affected")
    return tensor


def process_weather_events():
    """
    处理极端天气事件

    天气强度分级:
    - 0: 无事件
    - 1: 低档（Cold Weather, Very Hot Weather）
    - 2: 中档（Strong Monsoon, Amber级别的雨）
    - 3: 高档（其他所有警告）

    Returns:
        weather_tensor: (8760, 103) 标量张量（全域影响）
    """
    print("Processing extreme weather events...")

    tensor = np.zeros((NUM_HOURS, NUM_REGIONS), dtype=np.float32)

    # 读取天气警告数据
    weather_df = pd.read_csv(
        EVENTS_DIR / "weather_warnings_2019_processed.csv")

    # 定义天气强度映射
    def get_weather_intensity(warning_type, warning_title):
        """根据警告类型和标题返回强度等级"""
        warning_lower = warning_type.lower()
        title_lower = warning_title.lower()

        # 低档
        if 'cold weather' in warning_lower or 'cold weather' in title_lower:
            return 1.0
        if 'very hot' in warning_lower or 'very hot' in title_lower:
            return 1.0

        # 中档
        if 'strong monsoon' in warning_lower or 'strong monsoon' in title_lower:
            return 2.0
        if 'amber' in warning_lower or 'amber' in title_lower:
            if 'rain' in warning_lower or 'rain' in title_lower:
                return 2.0

        # 高档（其他所有类型）
        return 3.0

    for _, row in weather_df.iterrows():
        try:
            start_time = row['start_time']
            end_time = row['end_time']
            warning_type = row['warning_type']
            warning_title = row['warning_title']

            # 获取强度等级
            intensity = get_weather_intensity(warning_type, warning_title)

            # 计算时间索引
            start_hour = datetime_to_hour_index(start_time)

            # 对于天气事件，如果发生，则当天剩余时间都受影响
            dt = pd.to_datetime(start_time)
            end_of_day = datetime(dt.year, dt.month, dt.day, 23, 59, 59)
            end_hour = datetime_to_hour_index(
                end_of_day.strftime("%Y-%m-%d %H:%M"))

            # 确保索引在有效范围内
            start_hour = max(0, min(start_hour, NUM_HOURS - 1))
            end_hour = max(0, min(end_hour, NUM_HOURS - 1))

            # 天气事件影响所有区域（全域影响）
            # 如果同一时段有多个警告，取最大强度
            for h in range(start_hour, end_hour + 1):
                tensor[h, :] = np.maximum(tensor[h, :], intensity)

        except Exception as e:
            print(f"Error processing weather event: {e}, row: {row}")
            continue

    print(f"Weather events: {np.sum(tensor > 0)} hour-region pairs affected")
    print(f"  Level 1 (low): {np.sum(tensor == 1.0)}")
    print(f"  Level 2 (medium): {np.sum(tensor == 2.0)}")
    print(f"  Level 3 (high): {np.sum(tensor == 3.0)}")

    return tensor


def main():
    """主函数：生成时空事件张量"""
    print("=" * 60)
    print("Generating Spatio-Temporal Event Tensors for Hong Kong 2019")
    print("=" * 60)

    # 1. 加载映射数据
    print("\n[1/4] Loading station mappings and Voronoi regions...")
    station_map = load_station_mapping()
    gdf, code_to_idx = load_voronoi_regions()
    print(f"  Loaded {len(station_map)} station mappings")
    print(f"  Loaded {len(gdf)} Voronoi regions")

    # 2. 处理站点停运事件
    print("\n[2/4] Processing service outage events...")
    service_outage_tensor = process_service_outage(station_map, code_to_idx)

    # 3. 处理演唱会事件
    print("\n[3/4] Processing concert events...")
    concert_tensor = process_concerts(gdf)

    # 4. 处理天气事件
    print("\n[4/4] Processing weather events...")
    weather_tensor = process_weather_events()

    # 5. 保存结果
    print("\n[5/5] Saving event tensors...")
    np.savez_compressed(
        OUTPUT_FILE,
        service_outage=service_outage_tensor,
        concert=concert_tensor,
        extreme_weather=weather_tensor,
        description="HK 2019 Event Tensors: service_outage (binary), concert (binary), extreme_weather (0/1/2/3)",
        shape_info="(8760 hours, 103 regions)",
        year=YEAR
    )

    print(f"\n✓ Event tensors saved to: {OUTPUT_FILE}")
    print(f"  Shape: ({NUM_HOURS}, {NUM_REGIONS})")
    print(f"  Channels: 3 (service_outage, concert, extreme_weather)")

    # 统计信息
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    print(f"Service Outage Events:")
    print(
        f"  Total affected hours-regions: {np.sum(service_outage_tensor > 0)}")
    print(
        f"  Coverage: {np.sum(service_outage_tensor > 0) / (NUM_HOURS * NUM_REGIONS) * 100:.2f}%")

    print(f"\nConcert Events:")
    print(f"  Total affected hours-regions: {np.sum(concert_tensor > 0)}")
    print(
        f"  Coverage: {np.sum(concert_tensor > 0) / (NUM_HOURS * NUM_REGIONS) * 100:.2f}%")

    print(f"\nExtreme Weather Events:")
    print(f"  Total affected hours-regions: {np.sum(weather_tensor > 0)}")
    print(
        f"  Coverage: {np.sum(weather_tensor > 0) / (NUM_HOURS * NUM_REGIONS) * 100:.2f}%")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
