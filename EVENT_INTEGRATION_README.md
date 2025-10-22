# 事件数据整合到扩散模型 - 修改说明

## 概述

本次修改将时变事件张量 **E ∈ R^{N_l × T × d_event}** 整合到 KSTDiff 扩散模型中，作为额外的条件信息。现在模型的条件包括：

1. **静态区域特征 c**（原有）
2. **体量估计 s_hat**（原有）
3. **时变事件 E**（新增）

## 修改内容

### 1. 数据加载模块 (`load_data.py`)

#### 修改点：

- **`Data.__init__`**: 添加 `dataset_name` 参数，支持不同数据集的路径适配
- **`load_events()` 方法**: 新增事件张量加载功能

  - 从 `data/data_HK/Events/hk_events_2019.npz` 加载事件数据
  - 验证事件张量与区域数据的对齐
  - 如果没有事件数据，返回 `None`（使用零事件特征）

- **`load_pretrain()` 方法**: 修改以支持 HK 数据的特殊路径结构
  - HK 数据的 `ER.npz` 位于 `HK_regions/` 子目录

#### 数据结构：

```python
self.event_tensor: np.ndarray or None
    shape: (nreg, T, d_event)
    # nreg: 区域数量
    # T: 全年小时数（8760或8784）
    # d_event: 事件特征维度（通常为9）
```

### 2. 模型结构 (`model.py`)

#### 2.1 KGFlowBlock 修改

**初始化参数新增：**

- `event_dim`: 事件特征嵌入维度

**forward 方法新增参数：**

- `event_cond`: bs × nreg × nhour × event_emb_dim

**条件融合机制：**

```python
# 原始条件：静态特征
cond_static: nreg × cond_dim

# 时变事件条件
event_cond: bs × nreg × nhour × event_dim

# 融合方式：
# 1. 将静态条件扩展到每个时刻
# 2. 与事件条件在特征维度拼接
# 3. 通过1×1卷积投影到2*dim
cond_full = concat([cond_static_expanded, event_cond], dim=feature)
cond_proj = Conv1d(cond_full)  # -> (bs*nreg) × 2dim × nhour
```

#### 2.2 KGFlow 修改

**初始化新增：**

- `event_mlp`: 事件特征投影网络
  - 输入：d_event（原始事件特征维度，如 9）
  - 输出：event_emb_dim（可配置，默认 16）
  - 结构：Linear → ReLU → Linear

**forward 方法新增参数：**

- `event_features`: bs × nreg × nhour × d_event

**处理流程：**

1. 通过 `event_mlp` 将原始事件特征投影到嵌入空间
2. 传递给所有 KGFlowBlock 层
3. 在每个残差块中与静态条件融合

#### 2.3 GaussianDiffusion 修改

**初始化修改：**

- 存储事件张量数据
- 注册为 buffer，随模型移动到 GPU

**新增方法：**

```python
def extract_event_features(self, regids, hour_indices):
    """
    从事件张量中提取当前批次和时间段的事件特征

    Returns:
        event_features: bs × nreg × nhour × d_event
    """
```

**修改的方法：**

- `model_predictions()`: 提取事件特征并传递给模型
- `p_losses()`: 在训练时提取和传递事件特征

### 3. 训练脚本修改 (`main.py`, `main_sample.py`)

**命令行参数新增：**

```bash
--event_emb_dim: 事件嵌入维度（默认16）
```

**数据加载修改：**

```python
d = Data(data_dir=data_dir, dataset_name=args.dataset)
```

## 使用方法

### 1. 准备事件数据

确保事件张量文件存在：

```
data/data_HK/Events/hk_events_2019.npz
```

文件应包含：

- `event_tensor`: shape 为 (nreg, T, d_event) 的 numpy 数组

### 2. 训练模型

```bash
# HK数据集训练示例
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset HK \
    --event_emb_dim 16 \
    --num_iterations 2000 \
    --batch_size 2 \
    --lr 1e-5 \
    --n_layer 5 \
    --dim 64
```

### 3. 生成样本

```bash
CUDA_VISIBLE_DEVICES=0 python main_sample.py \
    --dataset HK \
    --event_emb_dim 16 \
    --model_path ./output/output_HK/model_1000.pth
```

## 关键设计说明

### 时变条件的处理

1. **静态 + 动态融合**：

   - 静态区域特征（KGE、scale）在所有时刻保持不变
   - 事件特征随时间变化，为每个时刻提供不同的条件信息

2. **特征投影**：

   - 原始事件特征（9 维）通过 MLP 投影到可学习的嵌入空间（16 维）
   - 这允许模型学习事件特征的最佳表示

3. **条件注入点**：
   - 在每个 KGFlowBlock 的残差连接之前
   - 通过 gate 机制（sigmoid + tanh）调制特征

### 事件特征的时间对齐

当前实现使用简化的时间对齐：

- 假设流量数据的 24 小时对应事件张量的固定时间段
- 未来改进：可以传入精确的时间戳进行动态对齐

### 向后兼容性

- 如果没有事件数据（`event_tensor=None`），模型自动退化为原始版本
- 已有数据集（NYC, DC, BM）可以继续使用，不受影响
- 事件特征维度为 0 时，条件融合模块自动跳过事件部分

## 实验配置建议

### 事件嵌入维度选择

| 事件特征原始维度 | 推荐嵌入维度 | 说明                                   |
| ---------------- | ------------ | -------------------------------------- |
| 9                | 16           | 默认配置，平衡性能和计算成本           |
| 9                | 32           | 更强的表达能力，适合事件影响显著的场景 |
| 9                | 8            | 轻量级，快速实验                       |

### 超参数调整

考虑到事件条件的加入，可能需要调整：

- **学习率**：可能需要略微降低（如从 1e-5 降至 5e-6）
- **条件模型更新频率**：`train_guidance_every_epochs` 可能需要更频繁
- **扩散步数**：可以尝试增加以利用更丰富的条件信息

## 验证和调试

### 检查事件数据加载

```python
from load_data import Data
d = Data(data_dir='./data/data_HK/', dataset_name='HK')
print(f"Event tensor shape: {d.event_tensor.shape}")
# 期望输出: Event tensor shape: (95, 8760, 9)
```

### 验证模型构建

```python
from model import KGFlow
model = KGFlow(d, event_emb_dim=16, ...)
print(f"Has events: {model.has_events}")
# 期望输出: Has events: True
```

### 监控训练

注意观察：

1. 训练 loss 是否正常下降
2. 条件预测模型的 RMSE
3. 生成样本的质量指标（MAE, RMSE, MMD）

## 文件修改清单

| 文件             | 修改类型 | 主要变更                                   |
| ---------------- | -------- | ------------------------------------------ |
| `load_data.py`   | 扩展     | 添加事件张量加载、HK 数据路径适配          |
| `model.py`       | 扩展     | 事件条件融合、时变条件处理                 |
| `main.py`        | 修改     | 添加 event_emb_dim 参数、传递 dataset_name |
| `main_sample.py` | 修改     | 传递 dataset_name 以加载事件数据           |

## 注意事项

1. **GPU 内存**：事件张量会被加载到 GPU，注意内存使用

   - 对于 95 个区域、8760 小时、9 维特征：约占用 ~30MB

2. **批次大小**：如果内存不足，可以减小 batch_size

3. **事件数据质量**：确保事件张量的归一化和预处理正确

   - 推荐范围：[0, 1] 或 [-1, 1]

4. **区域对齐**：确保事件数据的区域顺序与 `region2info.json` 一致

## 下一步扩展

可能的改进方向：

1. **动态时间对齐**：传入精确的时间戳，支持任意时间段
2. **事件注意力机制**：学习不同事件类型的重要性权重
3. **分层事件建模**：区分局部事件和全局事件
4. **事件预测**：将事件预测作为辅助任务

## 联系和反馈

如有问题或建议，请查看代码注释或联系开发团队。
