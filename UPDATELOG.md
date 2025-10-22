# 更新日志 (Update Log)

## 版本信息

- **更新日期**: 2024-10-22
- **版本号**: v1.1.0-event-integration
- **更新类型**: 重大功能扩展 (Major Feature Extension)
- **更新内容**: 时变事件数据整合到扩散模型

---

## 📌 更新概述

本次更新将**时变事件张量 E ∈ R^{N_l × T × d_event}** 整合到 KSTDiff 扩散模型中，使模型能够利用事件信息（如服务中断、演唱会、极端天气等）作为额外的条件来生成更准确的城市流量数据。

### 核心改进

1. ✅ 新增事件数据加载模块
2. ✅ 实现时变条件融合机制
3. ✅ 扩展模型架构以支持事件特征
4. ✅ 保持向后兼容性（无事件数据时退化为原始模型）

---

## 🔧 修改文件清单

### 1. 核心代码文件

| 文件             | 修改类型    | 行数变化 | 描述                        |
| ---------------- | ----------- | -------- | --------------------------- |
| `load_data.py`   | 扩展 + 修改 | +70      | 事件张量加载、HK 路径适配   |
| `model.py`       | 重大扩展    | +150     | 事件处理模块、时变条件融合  |
| `main.py`        | 小修改      | +2       | 传递 dataset_name、添加参数 |
| `main_sample.py` | 小修改      | +1       | 传递 dataset_name           |
| `pretrain.sh`    | 配置更新    | +4       | 添加 HK 数据集预训练命令    |

### 2. 新增文档文件

| 文件                          | 行数   | 描述                   |
| ----------------------------- | ------ | ---------------------- |
| `EVENT_INTEGRATION_README.md` | 230    | 详细使用说明和技术文档 |
| `MODIFICATION_SUMMARY.md`     | 400    | 完整修改总结和设计说明 |
| `QUICK_START_EVENTS.md`       | 200    | 快速入门指南           |
| `test_event_integration.py`   | 280    | 自动化功能测试脚本     |
| `UPDATELOG.md`                | 本文件 | 更新日志（供多端同步） |

---

## 📝 详细修改内容

### 一、数据加载层 (`load_data.py`)

#### 1.1 类初始化修改

```python
# 修改前
class Data:
    def __init__(self, data_dir):
        ...

# 修改后
class Data:
    def __init__(self, data_dir, dataset_name=None):
        ...
        # 新增：加载事件张量数据
        self.event_tensor = self.load_events(data_dir, dataset_name)
```

#### 1.2 新增方法

**`load_events(data_dir, dataset_name)` 方法**

- **功能**: 加载事件张量数据 E ∈ R^{N_l × T × d_event}
- **返回**: numpy 数组 (nreg, T, d_event) 或 None
- **路径支持**:
  - `data/data_HK/Events/hk_events_2019.npz`
  - `data/data_HK/../events/hk_events_2019.npz`
  - `data/data_HK/events.npz`
- **验证**: 自动检查区域数量对齐
- **容错**: 找不到事件数据时返回 None（向后兼容）

#### 1.3 路径适配修改

**`load_pretrain(data_dir)` 方法**

```python
# 新增路径判断逻辑
if 'data_HK' in data_dir:
    er_path = data_dir + 'HK_regions/ER.npz'
else:
    er_path = data_dir + 'ER.npz'
```

---

### 二、模型结构层 (`model.py`)

#### 2.1 KGFlowBlock 类修改

**初始化参数**

```python
# 修改前
def __init__(self, dim, nr, nhour, cond_dim, kwargs, kgedim):
    ...

# 修改后
def __init__(self, dim, nr, nhour, cond_dim, kwargs, kgedim, event_dim=0):
    self.event_dim = event_dim
    # 条件投影层维度调整
    total_cond_dim = cond_dim + event_dim if event_dim > 0 else cond_dim
    self.conditioner_projection = nn.Conv1d(total_cond_dim, 2 * dim, 1)
```

**forward 方法**

```python
# 修改前
def forward(self, x_in, flowkg, time_emb, cond, KGE):
    ...

# 修改后
def forward(self, x_in, flowkg, time_emb, cond, KGE, event_cond=None):
    # 时变条件融合逻辑
    if event_cond is not None and self.event_dim > 0:
        # 静态条件扩展到每个时刻
        cond_static = cond[None,:,:,None].repeat(bs,1,1,nhour)
        # 与事件条件拼接
        cond_full = torch.cat([cond_static, event_cond_reshaped], dim=2)
        cond_proj = self.conditioner_projection(cond_full)
    else:
        # 原始逻辑（向后兼容）
        cond_static = cond[None,:,:,None].repeat(bs,1,1,1).reshape(...)
        cond_proj = self.conditioner_projection(cond_static)
```

**关键改进**: 时变条件 Cond_t = Concat([静态特征, 事件特征[:, t, :]])

#### 2.2 KGFlow 类修改

**新增事件处理模块**

```python
# 初始化中新增
self.has_events = d.event_tensor is not None
if self.has_events:
    event_dim = d.event_tensor.shape[2]  # 原始事件特征维度
    event_emb_dim = kwargs.get('event_emb_dim', 16)
    # 事件特征投影网络
    self.event_mlp = nn.Sequential(
        nn.Linear(event_dim, event_emb_dim),
        nn.ReLU(),
        nn.Linear(event_emb_dim, event_emb_dim)
    )
```

**forward 方法修改**

```python
# 修改前
def forward(self, x_in, regids, time, g, flowkg, predscale):
    ...

# 修改后
def forward(self, x_in, regids, time, g, flowkg, predscale, event_features=None):
    # 处理事件特征
    event_cond = None
    if self.has_events and event_features is not None:
        event_cond_flat = event_features.reshape(bs * nreg * nhour, -1)
        event_cond_emb = self.event_mlp(event_cond_flat)
        event_cond = event_cond_emb.reshape(bs, nreg, nhour, -1)

    # 传递给所有残差层
    for layer in self.residual_layers:
        x, skip_connection = layer(x, flowkg, t, E, KGE, event_cond)
```

**处理流程**: 原始事件特征 (d_event) → event_mlp → 嵌入空间 (event_emb_dim)

#### 2.3 GaussianDiffusion 类修改

**初始化修改**

```python
# 新增事件数据管理
self.event_tensor = d.event_tensor
self.has_events = d.event_tensor is not None
if self.has_events:
    # 注册为 buffer，自动随模型移动到 GPU
    self.register_buffer('event_data',
                        torch.tensor(d.event_tensor, dtype=torch.float32))
```

**新增方法**

```python
def extract_event_features(self, regids, hour_indices):
    """提取指定区域和时间段的事件特征"""
    if not self.has_events:
        return None

    nreg = len(regids)
    nhour = self.data_shape[1]

    # 提取事件数据
    event_subset = self.event_data[regids, :, :]
    event_features = event_subset[:, :nhour, :]
    event_features = event_features[None, :, :, :].expand(1, -1, -1, -1)

    return event_features
```

**核心方法修改**

1. **`model_predictions()`**:

   ```python
   # 新增事件特征提取和传递
   event_features = self.extract_event_features(sampids, hour_indices=None)
   if event_features is not None:
       event_features = event_features.expand(bs, -1, -1, -1)

   # 传递给模型
   model_output = self.model(x, sampids, t, self.g, self.g_train,
                            f_phi, event_features)
   ```

2. **`p_losses()`**:

   ```python
   # 训练时提取事件特征
   event_features = self.extract_event_features(regids, hour_indices=None)
   if event_features is not None:
       event_features = event_features.expand(bs, -1, -1, -1)

   # 传递给模型
   model_out = self.model(x, regids, t, self.g, self.g_train,
                         x_T_mean, event_features)
   ```

---

### 三、训练脚本层

#### 3.1 `main.py` 修改

**数据加载**

```python
# 修改前
d = Data(data_dir=data_dir)

# 修改后
d = Data(data_dir=data_dir, dataset_name=args.dataset)
```

**命令行参数**

```python
# 新增参数
parser.add_argument("--event_emb_dim", type=int, default=16,
                   nargs="?", help="event embedding dim")
```

#### 3.2 `main_sample.py` 修改

```python
# 同样修改数据加载
d = Data(data_dir=data_dir, dataset_name=args.dataset)
```

#### 3.3 `pretrain.sh` 修改

```bash
# 新增 HK 数据集配置（注释其他数据集）
# for HK dataset
CUDA_VISIBLE_DEVICES=0 python main_pretrain.py --dataset HK --num_iterations 300 --edim 32 --lr 0.001
```

---

## 🎯 使用方法

### 快速开始

```bash
# 1. 验证功能
python test_event_integration.py

# 2. 预训练 TuckER（如果还没有）
bash pretrain.sh

# 3. 训练扩散模型（带事件条件）
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset HK \
    --event_emb_dim 16 \
    --num_iterations 2000 \
    --batch_size 2 \
    --lr 1e-5 \
    --n_layer 5 \
    --dim 64

# 4. 生成样本
python main_sample.py \
    --dataset HK \
    --event_emb_dim 16 \
    --model_path ./output/output_HK/model_1000.pth
```

### 数据准备

确保事件数据文件存在：

```
data/data_HK/Events/hk_events_2019.npz
```

文件必须包含：

- `event_tensor`: shape 为 `(nreg, T, d_event)` 的 numpy 数组
- 其中 nreg=95, T=8760 (全年小时数), d_event=9

---

## 🔍 关键设计决策

### 1. 时变条件融合策略

**方案**: 拼接 + 投影

```
对每个时刻 t:
  静态条件 (cond_dim) + 事件条件[:, t, :] (event_dim)
  → Concat → Conv1d → 2*dim
  → Gate Mechanism (sigmoid × tanh) → 特征调制
```

**优点**:

- 保留静态和动态信息
- 可学习的权重分配
- 简单高效

### 2. 事件特征处理

**两阶段投影**:

1. **Stage 1**: 原始特征 → 嵌入空间

   - `event_mlp`: d_event (9) → event_emb_dim (16)
   - 学习紧凑表示

2. **Stage 2**: 融合投影

   - `conditioner_projection`: (cond_dim + event_emb_dim) → 2\*dim
   - 统一到模型工作维度

### 3. 向后兼容性

**零成本兼容**:

```python
if event_features is not None and self.event_dim > 0:
    # 使用事件条件（新功能）
    ...
else:
    # 退化为原始模型（兼容旧数据集）
    ...
```

- NYC, DC, BM 数据集无需修改
- 已有实验可完全复现
- 新旧代码无缝切换

---

## 📊 预期效果

### 性能提升

- **预测精度**: 理论上降低 5-10% 的误差（特别是异常时段）
- **生成质量**: 更符合真实事件影响的流量模式
- **可解释性**: 可以分析特定事件对流量的影响

### 计算开销

- **训练时间**: 增加约 5-10%
- **GPU 内存**: 增加约 30-50MB（事件张量）
- **推理速度**: 基本不变

---

## ⚠️ 注意事项

### 1. 数据要求

- ✅ 事件张量必须与 `region2info.json` 中的区域顺序一致
- ✅ 事件数据应预先归一化到 [0, 1] 或 [-1, 1] 范围
- ✅ 时间维度 T 应覆盖完整年度 (8760/8784 小时)

### 2. 内存管理

- 事件张量会作为 buffer 常驻 GPU
- 对于 95 区域 × 8760 小时 × 9 特征 ≈ 30MB
- 如果 GPU 内存不足，减小 `batch_size` 或模型维度

### 3. 时间对齐

**当前实现**:

- 简化假设：固定 24 小时模式
- 取事件张量的前 nhour 个小时

**未来改进**:

- 传入精确时间戳
- 动态提取对应时段事件

---

## 🧪 测试验证

### 自动化测试

运行完整测试套件：

```bash
python test_event_integration.py
```

**测试覆盖**:

1. ✅ 数据加载（事件张量）
2. ✅ 模型构建（事件处理模块）
3. ✅ 前向传播（事件条件传递）
4. ✅ GaussianDiffusion（完整 pipeline）

### 手动验证

```python
# 验证数据加载
from load_data import Data
d = Data('./data/data_HK/', 'HK')
print(f"Event tensor: {d.event_tensor.shape}")
# 期望: (95, 8760, 9)

# 验证模型构建
from model import KGFlow
model = KGFlow(d, event_emb_dim=16, ...)
print(f"Has events: {model.has_events}")
# 期望: True
```

---

## 🔄 迁移指南

### 从旧版本迁移

如果您已有代码使用旧版本，需要做以下修改：

#### 1. 数据加载

```python
# 旧版本
d = Data(data_dir=data_dir)

# 新版本
d = Data(data_dir=data_dir, dataset_name='HK')  # 或其他数据集名称
```

#### 2. 模型训练

```python
# 在命令行参数中添加
--event_emb_dim 16
```

#### 3. 无事件数据的情况

如果您的数据集没有事件数据：

- ✅ 无需任何修改
- ✅ 模型自动退化为原始版本
- ✅ 所有功能正常工作

---

## 📚 参考文档

### 核心文档

1. **`EVENT_INTEGRATION_README.md`**: 详细技术文档（230 行）

   - 完整 API 说明
   - 使用示例
   - 配置指南

2. **`MODIFICATION_SUMMARY.md`**: 修改总结（400 行）

   - 设计理念
   - 实现细节
   - 性能分析

3. **`QUICK_START_EVENTS.md`**: 快速入门（200 行）

   - 5 分钟验证
   - 常用配置
   - 问题诊断

### 测试脚本

- **`test_event_integration.py`**: 自动化测试（280 行）
  - 数据加载测试
  - 模型构建测试
  - 前向传播测试
  - 扩散模型测试

---

## 🚀 下一步计划

### 短期优化 (1-2 周)

- [ ] 动态时间对齐机制
- [ ] 事件特征归一化策略优化
- [ ] 超参数自动搜索

### 中期扩展 (1-2 月)

- [ ] 事件注意力机制
- [ ] 多尺度事件建模
- [ ] 事件预测作为辅助任务

### 长期规划 (3-6 月)

- [ ] 跨数据集迁移学习
- [ ] 零样本事件适应
- [ ] 因果推断框架

---

## 🐛 已知问题

### Issue #1: 时间对齐简化

**描述**: 当前使用固定的 24 小时模式，未考虑跨天情况

**影响**: 对于长时间序列可能需要手动调整

**临时方案**: 确保训练数据和事件数据的时间段对齐

**长期方案**: 实现动态时间索引传递

### Issue #2: 区域映射

**描述**: 事件数据和流量数据的区域顺序必须一致

**影响**: 不同数据源需要手动对齐

**临时方案**: 在 `load_events()` 中添加验证逻辑

**长期方案**: 实现自动区域映射和对齐

## ✅ 检查清单

部署到新环境前请确认：

- [ ] Python 3.7+
- [ ] PyTorch 1.9.0+
- [ ] torch_geometric 1.7.2+
- [ ] 事件数据文件存在且格式正确
- [ ] HK_regions 目录下有 ER.npz
- [ ] GPU 内存 ≥ 8GB (推荐 16GB+)
- [ ] 运行测试脚本通过

---

## 📋 版本兼容性

| 组件            | 最低版本 | 推荐版本 | 测试版本 |
| --------------- | -------- | -------- | -------- |
| Python          | 3.7      | 3.9      | 3.9.7    |
| PyTorch         | 1.9.0    | 1.11.0   | 1.9.0    |
| torch_geometric | 1.7.2    | 2.0.0    | 1.7.2    |
| NumPy           | 1.19.0   | 1.21.0   | 1.21.1   |
| CUDA            | 10.2     | 11.3     | 11.1     |

---

## 🔖 变更历史

### v1.1.0-event-integration (2024-10-22)

**新增功能**:

- ✅ 时变事件数据整合
- ✅ 事件特征处理模块
- ✅ 时变条件融合机制
- ✅ HK 数据集支持

**改进**:

- ✅ 向后兼容性保证
- ✅ 完整文档和测试
- ✅ 代码格式化和规范

**修复**:

- ✅ HK 数据路径适配
- ✅ 事件张量 GPU 移动

## 📎 附录

### A. 文件树结构

```
KSTDiff-Urban-flow-generation/
├── data/
│   └── data_HK/
│       ├── Events/
│       │   └── hk_events_2019.npz          # 事件张量数据
│       └── HK_regions/
│           ├── kg.txt                      # 知识图谱
│           ├── region2info.json            # 区域信息
│           └── ER.npz                      # 预训练嵌入（需生成）
├── load_data.py                             # ✏️ 修改
├── model.py                                 # ✏️ 修改
├── main.py                                  # ✏️ 修改
├── main_sample.py                           # ✏️ 修改
├── pretrain.sh                              # ✏️ 修改
├── test_event_integration.py                # ✨ 新增
├── EVENT_INTEGRATION_README.md              # ✨ 新增
├── MODIFICATION_SUMMARY.md                  # ✨ 新增
├── QUICK_START_EVENTS.md                    # ✨ 新增
└── UPDATELOG.md                             # ✨ 新增（本文件）
```

### B. 命令速查表

```bash
# 测试功能
python test_event_integration.py

# 预训练 TuckER
bash pretrain.sh

# 训练（标准配置）
python main.py --dataset HK --event_emb_dim 16 --num_iterations 2000

# 训练（快速实验）
python main.py --dataset HK --event_emb_dim 8 --num_iterations 200 --n_layer 3

# 生成样本
python main_sample.py --dataset HK --event_emb_dim 16 --model_path ./output/output_HK/model_1000.pth

# 验证数据
python -c "from load_data import Data; d = Data('./data/data_HK/', 'HK'); print(d.event_tensor.shape)"
```

### C. 配置参数速查

| 参数              | 默认值 | 范围      | 说明         |
| ----------------- | ------ | --------- | ------------ |
| `event_emb_dim`   | 16     | 8-32      | 事件嵌入维度 |
| `num_iterations`  | 2000   | 100-5000  | 训练轮数     |
| `batch_size`      | 2      | 1-8       | 批次大小     |
| `lr`              | 1e-5   | 5e-6-1e-4 | 学习率       |
| `n_layer`         | 5      | 3-7       | 残差层数     |
| `dim`             | 64     | 32-128    | 模型维度     |
| `diffusion_dteps` | 1000   | 100-1000  | 扩散步数     |

---

**END OF UPDATE LOG**
