import os
import csv
import json
from glob import glob
from collections import defaultdict, OrderedDict

# 配置路径
DATA_DIR = "/Users/caffretro/Ca2+/HKU/HKU-Smart-Mobility-Lab/Data Center/Diffusion/Hong Kong/MTR OD/processed_hourly_data/"
REGION_INFO_PATH = "/Users/caffretro/Ca2+/HKU/HKU-Smart-Mobility-Lab/Development/KSTDiff-Urban-flow-generation/data/data_HK/HK_regions/region2info.json"
OUTPUT_JSON = os.path.join(DATA_DIR, "alldayflow.json")

# 规范：将 HOK/KOW/TSY 的机场快线(A)与非机场快线(T)合并为主站
MERGE_MAP = {
    "HOKA": "HOK", "HOKT": "HOK",
    "KOWA": "KOW", "KOWT": "KOW",
    "TSYA": "TSY", "TSYT": "TSY",
}


def norm_code(code: str) -> str:
    return MERGE_MAP.get(code, code)


# 读取 region2info.json 并确定区域顺序（按 key 的字典序排序）
with open(REGION_INFO_PATH, "r") as f:
    region2info = json.load(f)
regions_sorted = sorted(region2info.keys(), key=lambda x: x)
region_index = {r: i for i, r in enumerate(regions_sorted)}
nreg = len(regions_sorted)
nhour = 24

# 扫描 2019 年的小时 OD 文件
csv_files = sorted(glob(os.path.join(DATA_DIR, "hourly_od_2019*.csv")))

# 生成 {datestr: nreg x nhour x 2}，其中最后一维 [inflow, outflow]
date2flow = OrderedDict()

for path in csv_files:
    # 文件名形如 hourly_od_YYYYMMDD.csv
    base = os.path.basename(path)
    try:
        datestr = base.split("_")[2].split(".")[0]  # YYYYMMDD
        assert len(datestr) == 8 and datestr.isdigit()
    except Exception:
        # 跳过不规范的文件名
        continue

    # 初始化累计矩阵
    inflow = [[0.0 for _ in range(nhour)] for _ in range(nreg)]
    outflow = [[0.0 for _ in range(nhour)] for _ in range(nreg)]

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # 预期列：..., ORIGIN_CODE, ..., DESTINATION_CODE, ..., 0,1,...,23
        # 找到关键列索引
        try:
            o_idx = header.index("ORIGIN_CODE")
            d_idx = header.index("DESTINATION_CODE")
            hour_cols = [header.index(str(h)) for h in range(24)]
        except ValueError as e:
            raise RuntimeError(f"CSV header not as expected in {path}: {e}")

        for row in reader:
            o_raw = row[o_idx].strip()
            d_raw = row[d_idx].strip()
            o = norm_code(o_raw)
            d = norm_code(d_raw)

            # 非法站点或不在 region2info 中的直接跳过
            if o not in region_index or d not in region_index:
                continue

            o_i = region_index[o]
            d_i = region_index[d]

            # 聚合 24 个小时的客流
            for h, hc in enumerate(hour_cols):
                try:
                    v = float(row[hc]) if row[hc] != "" else 0.0
                except ValueError:
                    v = 0.0
                # outflow: 源站出发量；inflow: 目的站到达量
                outflow[o_i][h] += v
                inflow[d_i][h] += v

    # 组装 [nreg, nhour, 2]，最后一维 [inflow, outflow]
    mat = []
    for i in range(nreg):
        reg_hours = []
        for h in range(nhour):
            reg_hours.append([inflow[i][h], outflow[i][h]])
        mat.append(reg_hours)

    date2flow[datestr] = mat

# 写出 JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(date2flow, f, separators=(",", ":"))  # 与现有数据风格一致（紧凑）
print(f"Wrote {OUTPUT_JSON} with {len(date2flow)} day(s).")
