### compute_alldayflow_hk 脚本说明

#### 作用

- 从 MTR 小时 OD 数据（2019 年，每天一个 CSV，按小时列 0–23）聚合生成按 天 × 区域 × 小时 的 inflow/outflow，并写出 `alldayflow.json`。
- 合并 HOK/KOW/TSY 的机场快线(A)与非机场快线(T)拆分站名（如 `HOKA/HOKT` → `HOK`）。
- 输出格式符合 KSTDiff 规范：`{"YYYYMMDD": [nreg, nhour, 2]}`，最后一维为 `[inflow, outflow]`。

#### 输入

- 小时 OD CSV 文件（本目录）：文件名 `hourly_od_YYYYMMDD.csv`，仅处理以 `hourly_od_2019` 开头的文件。
- 列包含：`ORIGIN_CODE`, `DESTINATION_CODE`, 以及 24 个小时列：`0,1,...,23`。每行表示某日某起点到某终点的逐小时客流量。
- 区域信息：`region2info.json`（脚本中 `REGION_INFO_PATH` 指向香港版本）。

#### 区域顺序与站名合并

- 区域顺序：严格使用 `sorted(region2info.keys())` 的字典序，与 KSTDiff 加载逻辑一致。
- 站名合并（不区分机场快线/非机场快线）：
  - `HOKA`, `HOKT` → `HOK`
  - `KOWA`, `KOWT` → `KOW`
  - `TSYA`, `TSYT` → `TSY`
- 其他站名保持不变；若不在 `region2info.json` 中，跳过该行。

#### 聚合逻辑

- 对每条 (起点 o, 终点 d) 行的每个小时 h：
  - `outflow[o, h] += value`
  - `inflow[d, h] += value`
- 每日结束后，组装为 `[nreg, nhour, 2]`：每区域每小时为 `[inflow, outflow]`。

#### 输出

- 文件：`alldayflow.json`（写在本目录）。
- 结构：
  - 顶层键为日期字符串 `YYYYMMDD`；
  - 值为大小 `[nreg, nhour, 2]` 的三维数组（嵌套列表），最后一维 `[inflow, outflow]`。

#### 运行方式

- 确保脚本中的路径：
  - `DATA_DIR` 指向本目录；
  - `REGION_INFO_PATH` 指向香港 `region2info.json`。
- 运行（Python 3）：

```bash
python compute_alldayflow_hk.py
```

- 完成后将在本目录生成/覆盖 `alldayflow.json`，并打印写入天数。

#### 与 KSTDiff 的兼容性

- KSTDiff 的 `load_data.py` 会对 `alldayflow.json`：
  - 按键名 `YYYYMMDD` 过滤工作日；
  - 以所有工作日的 `min/max` 归一化到 [-1, 1]；
  - 假设区域顺序为 `sorted(region2info.keys())`（本脚本已保证）。

#### 注意

- CSV 中空值或非法数值按 0 处理；
- 异常命名文件将被跳过；
- 如需仅处理特定日期，可把 `glob("hourly_od_2019*.csv")` 改为单文件路径或增加筛选。
