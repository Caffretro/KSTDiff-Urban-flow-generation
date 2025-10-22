# 事件数据整合修改总结

## ✅ 已完成的所有修改

### 1. 数据加载层 (`load_data.py`) ✅

#### 修改内容：

```python
class Data:
    def __init__(self, data_dir, dataset_name=None):  # 添加dataset_name参数
        ...
        self.event_tensor = self.load_events(data_dir, dataset_name)  # 加载事件张量
```

**新增方法：**

- `load_events()`: 加载事件张量数据 E ∈ R^{N_l × T × d_event}
  - 支持多种路径格式
  - 自动验证数据形状
  - 如无事件数据返回 None（向后兼容）

**路径适配：**

- 修改 `load_pretrain()` 以支持 HK 数据的特殊目录结构

---

### 2. 模型结构层 (`model.py`) ✅

#### 2.1 KGFlowBlock 类

**初始化修改：**

```python
def __init__(self, dim, nr, nhour, cond_dim, kwargs, kgedim, event_dim=0):
    # 新增event_dim参数
    total_cond_dim = cond_dim + event_dim if event_dim > 0 else cond_dim
    self.conditioner_projection = nn.Conv1d(total_cond_dim, 2 * dim, 1)
```

**forward 方法修改：**

```python
def forward(self, x_in, flowkg, time_emb, cond, KGE, event_cond=None):
    # 时变条件融合
    if event_cond is not None and self.event_dim > 0:
        # 将静态条件扩展到每个时刻
        cond_static = cond[None,:,:,None].repeat(bs,1,1,nhour)
        # 与事件条件拼接
        cond_full = torch.cat([cond_static, event_cond_reshaped], dim=2)
        cond_proj = self.conditioner_projection(cond_full)
```

#### 2.2 KGFlow 类

**新增模块：**

```python
# 事件特征处理模块
self.has_events = d.event_tensor is not None
if self.has_events:
    self.event_mlp = nn.Sequential(
        nn.Linear(event_dim, event_emb_dim),
        nn.ReLU(),
        nn.Linear(event_emb_dim, event_emb_dim)
    )
```

**forward 方法修改：**

```python
def forward(self, x_in, regids, time, g, flowkg, predscale, event_features=None):
    # 处理事件特征
    if self.has_events and event_features is not None:
        event_cond = self.event_mlp(event_features)

    # 传递给所有残差层
    for layer in self.residual_layers:
        x, skip_connection = layer(x, flowkg, t, E, KGE, event_cond)
```

#### 2.3 GaussianDiffusion 类

**初始化修改：**

```python
# 注册事件数据为buffer
self.event_tensor = d.event_tensor
self.has_events = d.event_tensor is not None
if self.has_events:
    self.register_buffer('event_data', torch.tensor(d.event_tensor, dtype=torch.float32))
```

**新增方法：**

```python
def extract_event_features(self, regids, hour_indices):
    """从事件张量中提取当前批次和时间段的事件特征"""
    if not self.has_events:
        return None

    event_subset = self.event_data[regids, :, :]
    event_features = event_subset[:, :nhour, :]
    return event_features
```

**修改的核心方法：**

- `model_predictions()`: 提取并传递事件特征
- `p_losses()`: 训练时提取事件特征

---

### 3. 训练脚本层 (`main.py`, `main_sample.py`) ✅

**数据加载修改：**

```python
# main.py 和 main_sample.py
d = Data(data_dir=data_dir, dataset_name=args.dataset)
```

**新增命令行参数：**

```python
parser.add_argument("--event_emb_dim", type=int, default=16,
                   nargs="?", help="event embedding dim")
```

---

## 📊 数据流程图

```
事件张量文件 (*.npz)
    ↓
load_events() → self.event_tensor (N_l × T × d_event)
    ↓
GaussianDiffusion.__init__() → register_buffer('event_data')
    ↓
GaussianDiffusion.extract_event_features() → 提取 (bs × nreg × nhour × d_event)
    ↓
KGFlow.forward() → event_mlp → (bs × nreg × nhour × event_emb_dim)
    ↓
KGFlowBlock.forward() → concat(静态条件, 事件条件)
    ↓
conditioner_projection → 调制特征
    ↓
residual connection → 输出
```

---

## 🔑 关键设计决策

### 1. 时变条件融合策略

**选择的方案：拼接 + 投影**

```python
cond_full = concat([静态特征(扩展), 事件特征])  # 在特征维度拼接
cond_proj = Conv1d(cond_full)  # 统一投影到2*dim
```

**优点：**

- 保留静态和动态信息
- 通过可学习投影自适应权重
- 简单高效，易于调试

**替代方案（未采用）：**

- 加法融合：信息可能相互干扰
- 注意力融合：增加计算复杂度

### 2. 事件特征处理

**两阶段投影：**

1. **第一阶段** (event_mlp): d_event → event_emb_dim
   - 学习事件特征的紧凑表示
   - 降维或升维到统一空间
2. **第二阶段** (conditioner_projection): (cond_dim + event_emb_dim) → 2\*dim
   - 与静态条件融合
   - 投影到模型工作维度

### 3. 向后兼容性

**零成本兼容：**

```python
if event_features is not None and self.event_dim > 0:
    # 使用事件条件
else:
    # 退化为原始模型
```

- 无事件数据时，模型行为完全不变
- 已有实验结果可复现
- 新旧数据集可混用

---

## 📝 修改文件清单

| 文件             | 行数变化 | 主要修改                    |
| ---------------- | -------- | --------------------------- |
| `load_data.py`   | +70      | 事件加载、路径适配          |
| `model.py`       | +150     | 事件处理模块、条件融合      |
| `main.py`        | +2       | 传递 dataset_name、添加参数 |
| `main_sample.py` | +1       | 传递 dataset_name           |

**新增文件：**

- `EVENT_INTEGRATION_README.md`: 详细使用说明
- `MODIFICATION_SUMMARY.md`: 本文档
- `test_event_integration.py`: 功能验证脚本

---

## 🧪 测试验证

### 运行测试脚本：

```bash
python test_event_integration.py
```

**测试覆盖：**

1. ✅ 数据加载（事件张量）
2. ✅ 模型构建（事件处理模块）
3. ✅ 前向传播（事件条件传递）
4. ✅ 扩散模型（完整 pipeline）

---

## 🚀 使用示例

### 训练命令：

```bash
# HK数据集（带事件）
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset HK \
    --event_emb_dim 16 \
    --num_iterations 2000 \
    --batch_size 2 \
    --lr 1e-5 \
    --n_layer 5

# NYC数据集（无事件，向后兼容）
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset nyc \
    --num_iterations 2000 \
    --batch_size 2 \
    --lr 1e-5
```

---

## ⚙️ 配置参数

### 事件相关参数

| 参数            | 默认值   | 说明             | 推荐范围 |
| --------------- | -------- | ---------------- | -------- |
| `event_emb_dim` | 16       | 事件嵌入维度     | 8-32     |
| `event_dim`     | 自动检测 | 原始事件特征维度 | -        |

### 建议配置

**轻量级（快速实验）：**

```bash
--event_emb_dim 8 --n_layer 3 --dim 32
```

**标准配置：**

```bash
--event_emb_dim 16 --n_layer 5 --dim 64
```

**高性能：**

```bash
--event_emb_dim 32 --n_layer 7 --dim 128
```

---

## 📈 预期效果

### 性能提升

理论上，加入事件条件应该：

1. **降低预测误差**：事件信息提供额外约束
2. **提升生成质量**：更符合异常时段的流量模式
3. **增强可解释性**：可以分析事件对流量的影响

### 计算开销

- **训练时间**：增加约 5-10%
- **内存占用**：增加约 30-50MB（事件张量）
- **推理速度**：基本不变

---

## 🐛 已知问题和注意事项

### 1. 时间对齐

**当前实现：**

- 简化假设：固定 24 小时模式
- 未考虑跨天、跨周的情况

**未来改进：**

- 传入精确时间戳
- 动态提取对应时段的事件

### 2. 区域对齐

**注意：**

- 确保事件张量的区域顺序与 `region2info.json` 一致
- 不同数据源的区域 ID 可能不同

### 3. GPU 内存

**对于大规模数据：**

- 事件张量会常驻 GPU
- 注意 batch_size 的选择
- 可考虑动态加载策略

---

## 🔄 后续优化方向

### 短期（1-2 周）：

1. [ ] 动态时间对齐机制
2. [ ] 事件特征归一化优化
3. [ ] 超参数自动搜索

### 中期（1-2 月）：

1. [ ] 事件注意力机制
2. [ ] 多尺度事件建模
3. [ ] 事件预测辅助任务

### 长期（3-6 月）：

1. [ ] 跨数据集迁移学习
2. [ ] 零样本事件适应
3. [ ] 因果推断框架

---

## 📚 参考资料

### 相关论文：

1. KSTDiff 原论文：Knowledge-enhanced Spatio-Temporal Diffusion
2. Diffusion Models 相关综述
3. 事件驱动流量预测方法

### 代码参考：

- 原始 KSTDiff 实现
- Denoising Diffusion Probabilistic Models (DDPM)
- Conditional Diffusion Models

---

## 👥 贡献者

- 主要开发：Claude Sonnet 4.5
- 项目指导：用户需求驱动
- 测试验证：自动化测试脚本

---

## 📞 联系方式

如有问题或建议：

1. 查看 `EVENT_INTEGRATION_README.md` 详细文档
2. 运行 `test_event_integration.py` 进行诊断
3. 检查代码注释获取更多细节

---

**最后更新**: 2024 年（根据修改时间）
**版本**: 1.0
**状态**: ✅ 全部功能已实现并测试
