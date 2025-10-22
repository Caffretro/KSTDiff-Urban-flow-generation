# 事件整合快速入门指南

## 🎯 快速验证（5 分钟）

### 1. 测试数据加载

```bash
python -c "
from load_data import Data
d = Data('./data/data_HK/', 'HK')
print(f'✅ 区域数: {d.nreg}')
print(f'✅ 事件张量: {d.event_tensor.shape if d.event_tensor is not None else \"未找到\"}')
"
```

### 2. 运行完整测试

```bash
python test_event_integration.py
```

期望输出：

```
🚀🚀🚀...
测试1: 数据加载（包括事件张量）
   ✅ 数据加载成功
   ✅ 事件数据加载成功
...
🎉 所有测试通过！
```

---

## 🚀 开始训练（2 步）

### 第一步：预训练 TuckER（如果还没有）

```bash
bash pretrain.sh
```

等待输出：`ER.npz` 保存在 `data/data_HK/HK_regions/`

### 第二步：训练扩散模型（带事件）

```bash
# 修改 train.sh 或直接运行：
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset HK \
    --event_emb_dim 16 \
    --num_iterations 2000 \
    --batch_size 2 \
    --lr 1e-5 \
    --n_layer 5 \
    --dim 64
```

---

## 📊 监控训练

### 关键指标

**训练日志应显示：**

```
Loading data....
Event tensor registered: shape=torch.Size([95, 8760, 9])  ← 事件数据已加载
Event features enabled with dim=16                         ← 事件模块已启用
...
Epoch=1, train time cost ...s, loss:0.xxx
```

### MLflow 监控

```bash
# 如果配置了MLflow
mlflow ui --backend-store-uri ./mlflow_output/
# 访问 http://localhost:5000
```

---

## ⚙️ 常见配置

### 快速实验（降低计算成本）

```bash
python main.py \
    --dataset HK \
    --event_emb_dim 8 \
    --num_iterations 200 \
    --batch_size 2 \
    --n_layer 3 \
    --dim 32 \
    --diffusion_dteps 100
```

### 标准配置

```bash
python main.py \
    --dataset HK \
    --event_emb_dim 16 \
    --num_iterations 2000 \
    --batch_size 2 \
    --lr 1e-5 \
    --n_layer 5 \
    --dim 64
```

### 高性能配置（需要更多 GPU 内存）

```bash
python main.py \
    --dataset HK \
    --event_emb_dim 32 \
    --num_iterations 5000 \
    --batch_size 4 \
    --lr 5e-6 \
    --n_layer 7 \
    --dim 128
```

---

## 🎨 生成样本

训练完成后生成流量数据：

```bash
CUDA_VISIBLE_DEVICES=0 python main_sample.py \
    --dataset HK \
    --event_emb_dim 16 \
    --model_path ./output/output_HK/model_1000.pth \
    --sample_num 100
```

输出：`./output/output_HK/sample_final.npz`

---

## 🔍 问题诊断

### 如果事件数据未加载

```bash
# 检查文件是否存在
ls -lh data/data_HK/Events/hk_events_2019.npz

# 手动测试加载
python -c "
import numpy as np
data = np.load('data/data_HK/Events/hk_events_2019.npz')
print('Keys:', list(data.keys()))
print('Event tensor shape:', data['event_tensor'].shape)
"
```

### 如果 GPU 内存不足

```bash
# 减小batch_size
--batch_size 1

# 减小模型维度
--dim 32 --event_emb_dim 8 --n_layer 3

# 减少扩散步数
--diffusion_dteps 500
```

### 如果训练不稳定

```bash
# 降低学习率
--lr 5e-6

# 增加warmup（需修改代码）
# 或先训练条件模型
--pretrain_epochs 200
```

---

## 📈 评估结果

### 使用 evaluate.py

```bash
python evaluate.py \
    --generated_file ./output/output_HK/sample_final.npz \
    --ground_truth_file ./data/data_HK/test_data.npz
```

### 关键指标

- **MAE**: 平均绝对误差（越小越好）
- **RMSE**: 均方根误差（越小越好）
- **MMD**: 分布相似度（越小越好）

---

## 🔗 相关文档

- **详细说明**: `EVENT_INTEGRATION_README.md`
- **修改总结**: `MODIFICATION_SUMMARY.md`
- **测试脚本**: `test_event_integration.py`
- **代码流程**: `代码流程说明.md`

---

## ❓ 常见问题

### Q: 没有事件数据可以训练吗？

**A**: 可以！模型会自动退化为原始版本，不影响已有数据集。

### Q: 事件嵌入维度如何选择？

**A**:

- 原始事件维度 ≤ 16: 使用 16
- 原始事件维度 > 16: 使用 32
- 快速实验: 使用 8

### Q: 训练需要多久？

**A**:

- 2000 轮，batch_size=2: 约 6-10 小时（单 V100）
- 可以从 500 轮开始观察效果

### Q: 如何确认事件条件起作用？

**A**:

1. 查看训练日志：Event features enabled
2. 对比有/无事件的 loss 下降速度
3. 评估异常时段的预测精度

---

## 💡 最佳实践

1. **数据检查**：先运行 `test_event_integration.py`
2. **快速迭代**：从小模型、少 epoch 开始
3. **监控指标**：同时关注 loss 和生成质量
4. **保存中间结果**：每 100 轮保存一次模型
5. **对比实验**：训练有/无事件的两个版本

---

## 🎓 进阶使用

### 自定义事件特征

修改 `load_events()` 以加载自己的事件数据：

```python
def load_events(self, data_dir, dataset_name):
    # 加载你的事件数据
    event_tensor = your_loading_function()
    # 确保shape为 (nreg, T, d_event)
    return event_tensor
```

### 调整事件处理网络

修改 `model.py` 中的 `event_mlp`:

```python
self.event_mlp = nn.Sequential(
    nn.Linear(event_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_dim, event_emb_dim)
)
```

---

**准备好了吗？运行 `python test_event_integration.py` 开始！** 🚀
