"""
分析事件张量的稀疏性和覆盖量
计算每个通道每个月的覆盖量和稀疏性统计
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import calendar
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
BASE_DIR = Path(__file__).parent.parent
EVENTS_DIR = BASE_DIR / "data" / "data_HK" / "Events"
NPZ_FILE = EVENTS_DIR / "hk_events_2019.npz"

# 2019年参数
YEAR = 2019
NUM_HOURS = 24 * 365  # 8760小时
NUM_REGIONS = 95  # 实际区域数量（从数据中获取）

def load_event_tensors():
    """加载事件张量数据"""
    print("Loading event tensors...")
    data = np.load(NPZ_FILE)
    
    # 新的数据结构: 直接包含三个独立的张量
    service_outage = data['service_outage']
    concert_presence = data['concert']
    weather_intensity = data['extreme_weather']
    
    print(f"Service outage shape: {service_outage.shape}")
    print(f"Concert presence shape: {concert_presence.shape}")
    print(f"Weather intensity shape: {weather_intensity.shape}")
    print(f"Data type: {service_outage.dtype}")
    
    # 创建简化的metadata和statistics
    metadata = {
        'tensor_shape': [8760, 95],
        'dimensions': {'T': 8760, 'N_l': 95},
        'channels': {'service_outage': 0, 'concert': 1, 'extreme_weather': 2},
        'year': 2019
    }
    
    statistics = {
        'service_outage': {
            'nonzero_count': np.sum(service_outage > 0),
            'max_value': np.max(service_outage),
            'sparsity': 1 - (np.sum(service_outage > 0) / service_outage.size)
        },
        'concert': {
            'nonzero_count': np.sum(concert_presence > 0),
            'max_value': np.max(concert_presence),
            'sparsity': 1 - (np.sum(concert_presence > 0) / concert_presence.size)
        },
        'extreme_weather': {
            'nonzero_count': np.sum(weather_intensity > 0),
            'max_value': np.max(weather_intensity),
            'sparsity': 1 - (np.sum(weather_intensity > 0) / weather_intensity.size)
        }
    }
    
    return service_outage, concert_presence, weather_intensity, metadata, statistics

def get_month_hours():
    """获取2019年每个月的起始和结束小时索引"""
    month_info = []
    
    for month in range(1, 13):
        # 计算月初和月末
        start_date = pd.Timestamp(f'{YEAR}-{month:02d}-01')
        
        if month == 12:
            end_date = pd.Timestamp(f'{YEAR+1}-01-01') - pd.Timedelta(days=1)
        else:
            end_date = pd.Timestamp(f'{YEAR}-{month+1:02d}-01') - pd.Timedelta(days=1)
        
        # 计算小时索引
        start_hour = int((start_date - pd.Timestamp(f'{YEAR}-01-01')).total_seconds() / 3600)
        end_hour = int((end_date - pd.Timestamp(f'{YEAR}-01-01')).total_seconds() / 3600)
        
        month_info.append({
            'month': month,
            'month_name': calendar.month_name[month],
            'start_hour': start_hour,
            'end_hour': end_hour,
            'total_hours': end_hour - start_hour + 1
        })
    
    return month_info

def analyze_tensor_sparsity(tensor, tensor_name, month_info):
    """分析单个张量的稀疏性"""
    print(f"\n{'='*60}")
    print(f"Analyzing {tensor_name} tensor")
    print(f"{'='*60}")
    
    # 全局统计
    total_elements = tensor.size
    non_zero_elements = np.sum(tensor > 0)
    sparsity = 1 - (non_zero_elements / total_elements)
    
    print(f"Global Statistics:")
    print(f"  Total elements: {total_elements:,}")
    print(f"  Non-zero elements: {non_zero_elements:,}")
    print(f"  Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
    print(f"  Coverage: {(1-sparsity)*100:.2f}%")
    
    # 每月统计
    monthly_stats = []
    
    for month_data in month_info:
        month = month_data['month']
        month_name = month_data['month_name']
        start_hour = month_data['start_hour']
        end_hour = month_data['end_hour']
        
        # 提取该月的数据
        month_tensor = tensor[start_hour:end_hour+1, :]
        
        # 计算该月的统计
        month_total = month_tensor.size
        month_non_zero = np.sum(month_tensor > 0)
        month_sparsity = 1 - (month_non_zero / month_total)
        
        # 计算受影响的小时-区域对数量
        affected_hour_regions = month_non_zero
        
        # 计算平均每小时受影响区域数
        avg_regions_per_hour = month_non_zero / month_data['total_hours'] if month_data['total_hours'] > 0 else 0
        
        # 计算最大强度（对于天气事件）
        if tensor_name == 'extreme_weather':
            if month_tensor.size > 0:
                max_intensity = np.max(month_tensor)
                intensity_distribution = {
                    'level_0': np.sum(month_tensor == 0),
                    'level_1': np.sum(month_tensor == 1),
                    'level_2': np.sum(month_tensor == 2),
                    'level_3': np.sum(month_tensor == 3)
                }
            else:
                max_intensity = 0
                intensity_distribution = {'level_0': 0, 'level_1': 0, 'level_2': 0, 'level_3': 0}
        else:
            max_intensity = 1 if month_non_zero > 0 else 0
            intensity_distribution = None
        
        monthly_stat = {
            'month': month,
            'month_name': month_name,
            'total_hours': month_data['total_hours'],
            'total_elements': month_total,
            'non_zero_elements': month_non_zero,
            'sparsity': month_sparsity,
            'coverage': (1 - month_sparsity) * 100,
            'affected_hour_regions': affected_hour_regions,
            'avg_regions_per_hour': avg_regions_per_hour,
            'max_intensity': max_intensity,
            'intensity_distribution': intensity_distribution
        }
        
        monthly_stats.append(monthly_stat)
        
        coverage_str = f"{monthly_stat['coverage']:6.2f}" if not np.isnan(monthly_stat['coverage']) else "   N/A"
        print(f"{month_name:>12}: Coverage={coverage_str}%, "
              f"Affected={monthly_stat['affected_hour_regions']:6d}, "
              f"AvgRegions/Hour={monthly_stat['avg_regions_per_hour']:6.1f}")
    
    return monthly_stats

def create_visualizations(all_monthly_stats):
    """创建可视化图表"""
    print("\nCreating visualizations...")
    
    # 设置图表样式
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Event Tensor Analysis - Monthly Coverage and Sparsity', fontsize=16, fontweight='bold')
    
    # 准备数据
    months = [stat['month'] for stat in all_monthly_stats['service_outage']]
    month_names = [stat['month_name'] for stat in all_monthly_stats['service_outage']]
    
    # 1. 每月覆盖率对比
    ax1 = axes[0, 0]
    channels = ['service_outage', 'concert', 'extreme_weather']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, channel in enumerate(channels):
        coverage = [stat['coverage'] for stat in all_monthly_stats[channel]]
        ax1.plot(months, coverage, marker='o', linewidth=2, markersize=6, 
                label=channel.replace('_', ' ').title(), color=colors[i])
    
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Coverage (%)')
    ax1.set_title('Monthly Coverage by Channel')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(months)
    ax1.set_xticklabels([m[:3] for m in month_names], rotation=45)
    
    # 2. 每月稀疏性对比
    ax2 = axes[0, 1]
    for i, channel in enumerate(channels):
        sparsity = [stat['sparsity'] * 100 for stat in all_monthly_stats[channel]]
        ax2.plot(months, sparsity, marker='s', linewidth=2, markersize=6,
                label=channel.replace('_', ' ').title(), color=colors[i])
    
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Sparsity (%)')
    ax2.set_title('Monthly Sparsity by Channel')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(months)
    ax2.set_xticklabels([m[:3] for m in month_names], rotation=45)
    
    # 3. 受影响的小时-区域对数量
    ax3 = axes[1, 0]
    for i, channel in enumerate(channels):
        affected = [stat['affected_hour_regions'] for stat in all_monthly_stats[channel]]
        ax3.bar([m + i*0.25 for m in months], affected, width=0.25, 
               label=channel.replace('_', ' ').title(), color=colors[i], alpha=0.7)
    
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Affected Hour-Region Pairs')
    ax3.set_title('Monthly Affected Hour-Region Pairs')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks([m + 0.25 for m in months])
    ax3.set_xticklabels([m[:3] for m in month_names], rotation=45)
    
    # 4. 天气事件强度分布（仅显示extreme_weather）
    ax4 = axes[1, 1]
    weather_stats = all_monthly_stats['extreme_weather']
    
    # 计算每个强度等级的总数
    intensity_levels = ['Level 0', 'Level 1', 'Level 2', 'Level 3']
    intensity_totals = [0, 0, 0, 0]
    
    for stat in weather_stats:
        if stat['intensity_distribution']:
            intensity_totals[0] += stat['intensity_distribution']['level_0']
            intensity_totals[1] += stat['intensity_distribution']['level_1']
            intensity_totals[2] += stat['intensity_distribution']['level_2']
            intensity_totals[3] += stat['intensity_distribution']['level_3']
    
    bars = ax4.bar(intensity_levels, intensity_totals, 
                   color=['#E8E8E8', '#FFE066', '#FF8C42', '#FF4444'], alpha=0.8)
    ax4.set_xlabel('Weather Intensity Level')
    ax4.set_ylabel('Total Elements')
    ax4.set_title('Weather Event Intensity Distribution')
    ax4.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar, total in zip(bars, intensity_totals):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{total:,}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path(__file__).parent / "event_tensor_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    return fig

def save_detailed_report(all_monthly_stats):
    """保存详细的统计报告"""
    print("\nSaving detailed report...")
    
    # 创建DataFrame用于保存
    report_data = []
    
    for month_idx in range(12):
        month_name = all_monthly_stats['service_outage'][month_idx]['month_name']
        
        row = {
            'Month': month_name,
            'Service_Outage_Coverage': all_monthly_stats['service_outage'][month_idx]['coverage'],
            'Service_Outage_Sparsity': all_monthly_stats['service_outage'][month_idx]['sparsity'],
            'Service_Outage_Affected': all_monthly_stats['service_outage'][month_idx]['affected_hour_regions'],
            'Concert_Coverage': all_monthly_stats['concert'][month_idx]['coverage'],
            'Concert_Sparsity': all_monthly_stats['concert'][month_idx]['sparsity'],
            'Concert_Affected': all_monthly_stats['concert'][month_idx]['affected_hour_regions'],
            'Weather_Coverage': all_monthly_stats['extreme_weather'][month_idx]['coverage'],
            'Weather_Sparsity': all_monthly_stats['extreme_weather'][month_idx]['sparsity'],
            'Weather_Affected': all_monthly_stats['extreme_weather'][month_idx]['affected_hour_regions'],
        }
        
        # 添加天气强度分布
        weather_dist = all_monthly_stats['extreme_weather'][month_idx]['intensity_distribution']
        if weather_dist:
            row.update({
                'Weather_Level_0': weather_dist['level_0'],
                'Weather_Level_1': weather_dist['level_1'],
                'Weather_Level_2': weather_dist['level_2'],
                'Weather_Level_3': weather_dist['level_3']
            })
        
        report_data.append(row)
    
    # 保存为CSV
    df = pd.DataFrame(report_data)
    csv_path = Path(__file__).parent / "event_tensor_monthly_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"Detailed report saved to: {csv_path}")
    
    # 打印汇总统计
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for channel in ['service_outage', 'concert', 'extreme_weather']:
        stats = all_monthly_stats[channel]
        avg_coverage = np.mean([s['coverage'] for s in stats])
        avg_sparsity = np.mean([s['sparsity'] for s in stats])
        total_affected = sum([s['affected_hour_regions'] for s in stats])
        
        print(f"\n{channel.replace('_', ' ').title()}:")
        print(f"  Average Coverage: {avg_coverage:.2f}%")
        print(f"  Average Sparsity: {avg_sparsity:.4f} ({avg_sparsity*100:.2f}%)")
        print(f"  Total Affected Hour-Regions: {total_affected:,}")
        
        # 找出覆盖率和稀疏性的极值月份
        max_coverage_month = max(stats, key=lambda x: x['coverage'])
        min_coverage_month = min(stats, key=lambda x: x['coverage'])
        
        print(f"  Highest Coverage: {max_coverage_month['month_name']} ({max_coverage_month['coverage']:.2f}%)")
        print(f"  Lowest Coverage: {min_coverage_month['month_name']} ({min_coverage_month['coverage']:.2f}%)")

def main():
    """主函数"""
    print("="*80)
    print("Event Tensor Sparsity and Coverage Analysis")
    print("="*80)
    
    # 1. 加载数据
    service_outage, concert_presence, weather_intensity, metadata, statistics = load_event_tensors()
    
    # 2. 获取月份信息
    month_info = get_month_hours()
    
    # 3. 分析每个张量
    all_monthly_stats = {}
    
    all_monthly_stats['service_outage'] = analyze_tensor_sparsity(
        service_outage, 'service_outage', month_info)
    
    all_monthly_stats['concert'] = analyze_tensor_sparsity(
        concert_presence, 'concert', month_info)
    
    all_monthly_stats['extreme_weather'] = analyze_tensor_sparsity(
        weather_intensity, 'extreme_weather', month_info)
    
    # 4. 创建可视化
    fig = create_visualizations(all_monthly_stats)
    
    # 5. 保存详细报告
    save_detailed_report(all_monthly_stats)
    
    print("\n" + "="*80)
    print("Analysis completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
