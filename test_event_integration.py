#!/usr/bin/env python3
"""
测试事件数据整合到扩散模型的功能
验证数据加载、模型构建和forward传播
"""

from torch_geometric.data import Data as geoData
from model import KGFlow, GaussianDiffusion, DeterministicFeedForwardNeuralNetwork
from load_data import Data
import numpy as np
import torch
import sys
import os
sys.path.append('.')


def test_data_loading():
    """测试数据加载，包括事件张量"""
    print("=" * 60)
    print("测试1: 数据加载（包括事件张量）")
    print("=" * 60)

    try:
        data_dir = "./data/data_HK/"
        d = Data(data_dir=data_dir, dataset_name='HK')

        print(f"✅ 数据加载成功")
        print(f"   - 区域数量: {d.nreg}")
        print(f"   - 训练数据: {len(d.train_data)} days")
        print(f"   - 训练区域: {len(d.trainids)}")
        print(f"   - 测试区域: {len(d.sampids)}")

        if d.event_tensor is not None:
            print(f"   - 事件张量形状: {d.event_tensor.shape}")
            print(f"   - 事件特征维度: {d.event_tensor.shape[2]}")
            print(f"✅ 事件数据加载成功")
        else:
            print(f"⚠️  未找到事件数据（将使用零事件特征）")

        return d

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_construction(d):
    """测试模型构建，验证事件处理模块"""
    print("\n" + "=" * 60)
    print("测试2: 模型构建（包括事件处理模块）")
    print("=" * 60)

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        # 模型参数
        kwargs = {
            'dim': 64,
            'num_heads': 2,
            'num_rgcns': 1,
            'num_flowrgcns': 1,
            'num_sas': 1,
            'dropout': 0.0,
            'kge_cat_dim': 16,
            'xt_cat_dim': 16,
            'n_layer': 2,  # 使用较少层数以加快测试
            'event_emb_dim': 16,  # 事件嵌入维度
            'device': device
        }

        # 构建模型
        model = KGFlow(d=d, **kwargs)
        model = model.to(device)

        print(f"✅ KGFlow模型构建成功")
        print(f"   - 事件处理模块: {'启用' if model.has_events else '禁用'}")
        if model.has_events:
            print(f"   - 事件嵌入维度: {kwargs['event_emb_dim']}")
        print(f"   - 残差层数: {len(model.residual_layers)}")

        # 构建条件预测模型
        nn_x = torch.tensor([x[0] for x in d.scale_pred_data], device=device)
        dim_in = nn_x.shape[1]
        cond_pred_model = DeterministicFeedForwardNeuralNetwork(
            dim_in=dim_in, dim_out=1
        ).to(device)

        print(f"✅ 条件预测模型构建成功")

        return model, cond_pred_model, kwargs

    except Exception as e:
        print(f"❌ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_forward_pass(d, model, cond_pred_model, kwargs):
    """测试前向传播，包括事件条件"""
    print("\n" + "=" * 60)
    print("测试3: 前向传播（包括事件条件）")
    print("=" * 60)

    try:
        device = next(model.parameters()).device

        # 构建图结构
        trainids = d.trainids
        edge_index = torch.tensor(
            [[x[0] for x in d.trainkg_data], [x[2] for x in d.trainkg_data]],
            dtype=torch.long, device=device
        )
        edge_type = torch.tensor(
            [x[1] for x in d.trainkg_data],
            dtype=torch.int, device=device
        )
        g_train = geoData(edge_index=edge_index, edge_type=edge_type)

        # 准备测试数据
        bs = 2  # batch size
        nreg = len(trainids)
        nhour = len(d.train_data[0][0])

        # 随机生成测试输入
        x_in = torch.randn(bs, nreg, nhour, 2, device=device)
        time = torch.randint(0, 1000, (bs,), device=device).long()
        regids = torch.tensor(trainids, device=device)

        # 预测scale
        predscale = x_in.clone()

        # 生成事件特征（如果有事件数据）
        event_features = None
        if model.has_events:
            d_event = d.event_tensor.shape[2]
            # 模拟事件特征（实际会从event_tensor提取）
            event_features = torch.randn(
                bs, nreg, nhour, d_event, device=device)
            print(f"   - 使用事件特征: shape={event_features.shape}")

        # 前向传播
        print("   - 执行前向传播...")
        with torch.no_grad():
            output = model(
                x_in, regids, time, None, g_train, predscale, event_features
            )

        print(f"✅ 前向传播成功")
        print(f"   - 输入形状: {x_in.shape}")
        print(f"   - 输出形状: {output.shape}")
        print(
            f"   - 输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")

        return True

    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_diffusion_model(d, model, cond_pred_model, kwargs):
    """测试完整的扩散模型"""
    print("\n" + "=" * 60)
    print("测试4: GaussianDiffusion模型")
    print("=" * 60)

    try:
        device = next(model.parameters()).device

        # 构建图结构
        trainids = d.trainids

        # 完整KG
        edge_index_full = torch.tensor(
            [[x[0] for x in d.kg_data], [x[2] for x in d.kg_data]],
            dtype=torch.long, device=device
        )
        edge_type_full = torch.tensor(
            [x[1] for x in d.kg_data],
            dtype=torch.int, device=device
        )
        g = geoData(edge_index=edge_index_full, edge_type=edge_type_full)

        # 训练KG
        edge_index_train = torch.tensor(
            [[x[0] for x in d.trainkg_data], [x[2] for x in d.trainkg_data]],
            dtype=torch.long, device=device
        )
        edge_type_train = torch.tensor(
            [x[1] for x in d.trainkg_data],
            dtype=torch.int, device=device
        )
        g_train = geoData(edge_index=edge_index_train,
                          edge_type=edge_type_train)

        # 采样KG
        edge_index_samp = torch.tensor(
            [[x[0] for x in d.samplekg_data], [x[2] for x in d.samplekg_data]],
            dtype=torch.long, device=device
        )
        edge_type_samp = torch.tensor(
            [x[1] for x in d.samplekg_data],
            dtype=torch.int, device=device
        )
        g_samp = geoData(edge_index=edge_index_samp, edge_type=edge_type_samp)

        # 构建扩散模型
        data_shape = (len(d.train_data[0]), len(
            d.train_data[0][0]), len(d.train_data[0][0][0]))

        diffusion = GaussianDiffusion(
            model,
            cond_pred_model=cond_pred_model,
            d=d,
            data_shape=data_shape,
            g=g,
            g_train=g_train,
            g_samp=g_samp,
            image_size=128,
            beta_schedule='cosine',
            timesteps=100,  # 使用较少步数以加快测试
            loss_type='l1',
            objective='pred_noise'
        )
        diffusion = diffusion.to(device)

        print(f"✅ GaussianDiffusion模型构建成功")
        if diffusion.has_events:
            print(f"   - 事件数据已注册到模型")
            print(f"   - 事件buffer形状: {diffusion.event_data.shape}")

        # 测试训练步骤
        print("\n   - 测试训练步骤...")
        bs = 1
        nreg = len(trainids)
        nhour = data_shape[1]

        x_batch = torch.randn(bs, nreg, nhour, 2, device=device)
        regids = torch.tensor(trainids, device=device)

        with torch.no_grad():
            loss = diffusion(x_batch, regids)

        print(f"✅ 训练步骤测试成功")
        print(f"   - 损失值: {loss.item():.6f}")

        return True

    except Exception as e:
        print(f"❌ GaussianDiffusion测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "🚀" * 30)
    print("开始测试事件数据整合功能")
    print("🚀" * 30 + "\n")

    # 测试1: 数据加载
    d = test_data_loading()
    if d is None:
        print("\n❌ 测试失败：数据加载阶段出错")
        return

    # 测试2: 模型构建
    model, cond_pred_model, kwargs = test_model_construction(d)
    if model is None:
        print("\n❌ 测试失败：模型构建阶段出错")
        return

    # 测试3: 前向传播
    success = test_forward_pass(d, model, cond_pred_model, kwargs)
    if not success:
        print("\n❌ 测试失败：前向传播阶段出错")
        return

    # 测试4: 扩散模型
    success = test_diffusion_model(d, model, cond_pred_model, kwargs)
    if not success:
        print("\n❌ 测试失败：扩散模型阶段出错")
        return

    print("\n" + "=" * 60)
    print("🎉 所有测试通过！")
    print("=" * 60)
    print("\n📝 事件数据整合功能验证完成！")
    print("\n接下来可以运行完整训练：")
    print("   bash train.sh")
    print("   或")
    print("   python main.py --dataset HK --event_emb_dim 16")


if __name__ == "__main__":
    main()
