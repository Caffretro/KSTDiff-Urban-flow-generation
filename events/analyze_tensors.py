#!/usr/bin/env python3
"""
Script to load and analyze generated event tensors.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_event_tensors(file_path):
    """Load event tensors from .npz file."""
    try:
        data = np.load(file_path, allow_pickle=True)
        
        tensors = {
            'event_tensor': data['event_tensor'],
            'metadata': json.loads(str(data['metadata'])),
            'statistics': json.loads(str(data['statistics']))
        }
        
        print(f"Successfully loaded event tensors from: {file_path}")
        return tensors
    
    except Exception as e:
        print(f"Error loading tensors: {e}")
        return None

def analyze_tensor_structure(tensors):
    """Analyze the structure and content of the event tensor."""
    event_tensor = tensors['event_tensor']
    metadata = tensors['metadata']
    
    print("\n" + "="*60)
    print("TENSOR STRUCTURE ANALYSIS")
    print("="*60)
    
    print(f"Tensor shape: {event_tensor.shape}")
    print(f"Data type: {event_tensor.dtype}")
    print(f"Memory usage: {event_tensor.nbytes / 1024**2:.2f} MB")
    
    # Dimensions
    N_l, T, d_event = event_tensor.shape
    print(f"\nDimensions:")
    print(f"  N_l (regions): {N_l}")
    print(f"  T (timesteps): {T}")  
    print(f"  d_event (channels): {d_event}")
    
    # Channel descriptions
    print(f"\nChannel descriptions:")
    for ch, desc in metadata['channel_descriptions'].items():
        print(f"  Channel {ch}: {desc}")
    
    return event_tensor.shape

def analyze_tensor_statistics(tensors):
    """Analyze tensor statistics and sparsity."""
    event_tensor = tensors['event_tensor']
    statistics = tensors['statistics']
    
    print("\n" + "="*60)
    print("TENSOR STATISTICS")
    print("="*60)
    
    # Overall statistics
    tensor_stats = statistics['tensor_statistics']
    print(f"Total non-zero entries: {tensor_stats['total_nonzero_entries']:,}")
    print(f"Sparsity: {tensor_stats['sparsity']:.4f} ({tensor_stats['sparsity']*100:.2f}%)")
    print(f"Active regions: {tensor_stats['active_regions']}/{event_tensor.shape[0]}")
    print(f"Active timesteps: {tensor_stats['active_timesteps']}/{event_tensor.shape[1]}")
    print(f"Maximum value: {tensor_stats['max_value']:.4f}")
    print(f"Mean non-zero value: {tensor_stats['mean_nonzero']:.4f}")
    
    # Channel-wise statistics
    print(f"\nChannel-wise statistics:")
    channel_stats = statistics['channel_statistics']
    
    channel_names = [
        "Service Outage", "Service Impact", "Disruption Intensity",
        "Concert Presence", "Concert Intensity", "Attendance Level", 
        "Weather Intensity", "Weather Presence", "Weather Coverage"
    ]
    
    for ch in range(len(channel_names)):
        if str(ch) in channel_stats:
            stats = channel_stats[str(ch)]
            print(f"  {ch:2d} - {channel_names[ch]:18s}: "
                  f"{stats['nonzero_count']:6,} entries, "
                  f"max={stats['max_value']:.3f}, "
                  f"sparsity={stats['sparsity']:.4f}")

def visualize_tensor_overview(tensors, save_path=None):
    """Create overview visualizations of the event tensor."""
    event_tensor = tensors['event_tensor']
    N_l, T, d_event = event_tensor.shape
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Set up the plot
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Channel activity over time (sum across all regions)
    plt.subplot(3, 3, 1)
    time_activity = np.sum(event_tensor, axis=0)  # Sum over regions
    
    for ch in range(min(d_event, 5)):  # Show first 5 channels
        plt.plot(time_activity[:, ch], label=f'Channel {ch}', alpha=0.7)
    
    plt.title('Channel Activity Over Time')
    plt.xlabel('Time (hours)')
    plt.ylabel('Total Activity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Regional activity (sum across time and channels)
    plt.subplot(3, 3, 2)
    region_activity = np.sum(event_tensor, axis=(1, 2))
    plt.bar(range(N_l), region_activity)
    plt.title('Total Activity by Region')
    plt.xlabel('Region Index')
    plt.ylabel('Total Activity')
    plt.grid(True, alpha=0.3)
    
    # 3. Channel activity heatmap (region vs time, first channel)
    plt.subplot(3, 3, 3)
    # Sample every 24 hours to make it manageable
    sampled_data = event_tensor[:, ::24, 0]  # Every day, channel 0
    plt.imshow(sampled_data, aspect='auto', cmap='viridis')
    plt.title('Service Outage (Channel 0)\nRegions vs Time (daily)')
    plt.xlabel('Days')
    plt.ylabel('Region')
    plt.colorbar()
    
    # 4. Service events (channels 0-2)
    plt.subplot(3, 3, 4)
    service_data = event_tensor[:, :, 0:3]
    daily_service = np.sum(service_data, axis=0)[::24]  # Daily aggregation
    
    for ch in range(3):
        plt.plot(daily_service[:, ch], label=f'Service Ch {ch}', alpha=0.8)
    
    plt.title('Service Events (Daily)')
    plt.xlabel('Days')
    plt.ylabel('Activity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Concert events (channels 3-5)
    plt.subplot(3, 3, 5)
    concert_data = event_tensor[:, :, 3:6]
    daily_concert = np.sum(concert_data, axis=0)[::24]  # Daily aggregation
    
    for ch in range(3):
        plt.plot(daily_concert[:, ch], label=f'Concert Ch {ch+3}', alpha=0.8)
    
    plt.title('Concert Events (Daily)')
    plt.xlabel('Days')
    plt.ylabel('Activity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Weather events (channels 6-8)
    plt.subplot(3, 3, 6)
    weather_data = event_tensor[:, :, 6:9]
    daily_weather = np.sum(weather_data, axis=0)[::24]  # Daily aggregation
    
    for ch in range(3):
        plt.plot(daily_weather[:, ch], label=f'Weather Ch {ch+6}', alpha=0.8)
    
    plt.title('Weather Events (Daily)')
    plt.xlabel('Days')
    plt.ylabel('Activity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Monthly aggregation
    plt.subplot(3, 3, 7)
    # Aggregate by month (30 days = 720 hours)
    hours_per_month = 720
    n_months = T // hours_per_month
    
    monthly_totals = []
    for month in range(n_months):
        start_idx = month * hours_per_month
        end_idx = min(start_idx + hours_per_month, T)
        month_total = np.sum(event_tensor[:, start_idx:end_idx, :])
        monthly_totals.append(month_total)
    
    plt.bar(range(len(monthly_totals)), monthly_totals)
    plt.title('Total Event Activity by Month')
    plt.xlabel('Month')
    plt.ylabel('Total Activity')
    plt.grid(True, alpha=0.3)
    
    # 8. Channel correlation heatmap
    plt.subplot(3, 3, 8)
    # Flatten spatial-temporal dimensions for correlation
    reshaped = event_tensor.reshape(-1, d_event)
    correlation_matrix = np.corrcoef(reshaped.T)
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Channel Correlation Matrix')
    
    # 9. Sparsity by channel
    plt.subplot(3, 3, 9)
    sparsities = []
    for ch in range(d_event):
        channel_data = event_tensor[:, :, ch]
        sparsity = 1.0 - np.count_nonzero(channel_data) / channel_data.size
        sparsities.append(sparsity)
    
    plt.bar(range(d_event), sparsities)
    plt.title('Sparsity by Channel')
    plt.xlabel('Channel')
    plt.ylabel('Sparsity')
    plt.ylim(0, 1)
    for i, v in enumerate(sparsities):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def extract_sample_data(tensors, n_samples=10):
    """Extract some sample data points for inspection."""
    event_tensor = tensors['event_tensor']
    
    print("\n" + "="*60)
    print("SAMPLE DATA EXTRACTION")
    print("="*60)
    
    # Find non-zero entries
    nonzero_indices = np.nonzero(event_tensor)
    total_nonzero = len(nonzero_indices[0])
    
    print(f"Total non-zero entries: {total_nonzero:,}")
    
    if total_nonzero > 0:
        # Sample random non-zero entries
        sample_indices = np.random.choice(total_nonzero, 
                                        min(n_samples, total_nonzero), 
                                        replace=False)
        
        print(f"\nSample non-zero entries:")
        print("Region | Time | Channel | Value")
        print("-" * 35)
        
        for idx in sample_indices:
            region = nonzero_indices[0][idx]
            time = nonzero_indices[1][idx]  
            channel = nonzero_indices[2][idx]
            value = event_tensor[region, time, channel]
            
            print(f"{region:6d} | {time:4d} | {channel:7d} | {value:5.3f}")

def time_analysis(tensors):
    """Analyze temporal patterns in the data."""
    event_tensor = tensors['event_tensor']
    
    print("\n" + "="*60)
    print("TEMPORAL PATTERN ANALYSIS") 
    print("="*60)
    
    # Total activity by hour of day
    N_l, T, d_event = event_tensor.shape
    
    # Aggregate by hour of day (0-23)
    hourly_pattern = np.zeros(24)
    for hour in range(24):
        # Get all timesteps corresponding to this hour
        hour_indices = [i for i in range(hour, T, 24)]
        hourly_activity = np.sum(event_tensor[:, hour_indices, :])
        hourly_pattern[hour] = hourly_activity
    
    peak_hour = np.argmax(hourly_pattern)
    print(f"Peak activity hour: {peak_hour}:00")
    print(f"Peak activity value: {hourly_pattern[peak_hour]:.2f}")
    
    # Day of week pattern (assuming starting from Monday)
    weekly_pattern = np.zeros(7)
    for day in range(7):
        day_indices = [i for i in range(day * 24, T, 7 * 24)]
        if day_indices:
            weekly_activity = np.sum(event_tensor[:, day_indices, :])
            weekly_pattern[day] = weekly_activity
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    peak_day = np.argmax(weekly_pattern)
    print(f"Peak activity day: {days[peak_day]}")

def main():
    """Main function to analyze event tensors."""
    
    # File path - adjust as needed
    tensor_file = "hk_events_2019.npz"
    
    if not os.path.exists(tensor_file):
        print(f"Error: Tensor file not found: {tensor_file}")
        print("Please run the event tensor generation first.")
        return
    
    # Load tensors
    tensors = load_event_tensors(tensor_file)
    if tensors is None:
        return
    
    # Perform analysis
    analyze_tensor_structure(tensors)
    analyze_tensor_statistics(tensors)
    extract_sample_data(tensors)
    time_analysis(tensors)
    
    # Generate visualizations
    viz_file = "event_tensor_analysis.png"
    visualize_tensor_overview(tensors, viz_file)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Visualization saved as: {viz_file}")
    
    # Additional usage examples
    print(f"\nUsage examples:")
    print(f"# Load in your code:")
    print(f"import numpy as np")
    print(f"data = np.load('{tensor_file}', allow_pickle=True)")
    print(f"event_tensor = data['event_tensor']  # Shape: {tensors['event_tensor'].shape}")
    print(f"")
    print(f"# Access specific data:")
    print(f"service_outages = event_tensor[:, :, 0]     # Service outage binary")
    print(f"concert_events = event_tensor[:, :, 3:6]    # All concert channels") 
    print(f"weather_events = event_tensor[:, :, 6:9]    # All weather channels")

if __name__ == "__main__":
    main()