#!/usr/bin/env python3
"""
Visualization utilities for the hk_events_2019.npz tensor.

Generates:
- channel_sparsity.png: bar chart of nonzero counts and sparsity per channel
- temporal_aggregates_channel_*.png: daily aggregate time series per active channel
- heatmap_most_active_day_channel_*.png: heatmap (regions x 24 hours) for the most active day per channel

Usage:
  python events/visualize_event_tensor.py \
      --npz events/hk_events_2019.npz \
      --outdir outputs/event_visualizations

Options allow selecting specific channels and saving CSV summaries.
"""

import argparse
import json
import os
from typing import Dict, Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt


def ensure_outdir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_npz_with_metadata(npz_path: str) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str], Dict[str, int]]:
    data = np.load(npz_path, allow_pickle=True)

    if "event_tensor" not in data.files:
        raise ValueError("NPZ missing 'event_tensor' key")
    event_tensor: np.ndarray = data["event_tensor"]

    # metadata could be stored as 0-dim ndarray of string JSON
    channel_index_to_name: Dict[int, str] = {}
    dims: Dict[str, int] = {"N_l": event_tensor.shape[0],
                            "T": event_tensor.shape[1], "d_event": event_tensor.shape[2]}
    channels_name_to_index: Dict[str, int] = {}

    if "metadata" in data.files:
        meta_arr = data["metadata"]
        try:
            if isinstance(meta_arr, np.ndarray) and meta_arr.shape == ():
                metadata = json.loads(str(meta_arr.item()))
            else:
                metadata = json.loads(str(meta_arr))
        except Exception:
            metadata = {}

        if isinstance(metadata, dict):
            # prefer explicit mapping if present
            channels = metadata.get("channels")
            if isinstance(channels, dict):
                for name, idx in channels.items():
                    # keys may be strings; normalize to int-index mapping and name mapping
                    if isinstance(idx, int):
                        channel_index_to_name[idx] = str(name)
                channels_name_to_index = {v: k for k,
                                          v in channel_index_to_name.items()}

            dims_meta = metadata.get("dimensions")
            if isinstance(dims_meta, dict):
                for k in ("N_l", "T", "d_event"):
                    if k in dims_meta and isinstance(dims_meta[k], int):
                        dims[k] = dims_meta[k]

    # fallback channel names if not found
    if not channel_index_to_name:
        channel_index_to_name = {
            i: f"channel_{i}" for i in range(event_tensor.shape[2])}
        channels_name_to_index = {v: k for k,
                                  v in channel_index_to_name.items()}

    return event_tensor, dims, channel_index_to_name, channels_name_to_index


def compute_channel_stats(event_tensor: np.ndarray) -> Dict[int, Dict[str, float]]:
    stats: Dict[int, Dict[str, float]] = {}
    N_l, T, C = event_tensor.shape
    total = N_l * T
    for c in range(C):
        channel_data = event_tensor[:, :, c]
        nonzero = int(np.count_nonzero(channel_data))
        max_val = float(channel_data.max(initial=0.0))
        mean_nonzero = float(
            channel_data[channel_data > 0].mean()) if nonzero > 0 else 0.0
        sparsity = 1.0 - (nonzero / float(total))
        stats[c] = {
            "nonzero_count": nonzero,
            "max_value": max_val,
            "mean_nonzero": mean_nonzero,
            "sparsity": sparsity,
        }
    return stats


def plot_channel_sparsity(stats: Dict[int, Dict[str, float]], channel_index_to_name: Dict[int, str], outdir: str) -> str:
    indices = sorted(stats.keys())
    names = [channel_index_to_name.get(i, f"ch_{i}") for i in indices]
    nonzeros = [stats[i]["nonzero_count"] for i in indices]
    sparsities = [stats[i]["sparsity"] for i in indices]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    color1 = "tab:blue"
    ax1.set_xlabel("Channel")
    ax1.set_ylabel("Nonzero count", color=color1)
    bars = ax1.bar(names, nonzeros, color=color1, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticklabels(names, rotation=30, ha='right')

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Sparsity", color=color2)
    ax2.plot(names, sparsities, color=color2, marker='o')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1.0)

    fig.tight_layout()
    outpath = os.path.join(outdir, "channel_sparsity.png")
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def aggregate_daily(event_tensor: np.ndarray) -> np.ndarray:
    N_l, T, C = event_tensor.shape
    hours_per_day = 24
    days = T // hours_per_day
    trimmed = event_tensor[:, :days * hours_per_day, :]
    reshaped = trimmed.reshape(N_l, days, hours_per_day, C)
    # sum over hours then regions to get a daily total per channel
    daily = reshaped.sum(axis=2).sum(axis=0)  # shape: (days, C)
    return daily


def plot_temporal_aggregates_daily(event_tensor: np.ndarray, channel_index_to_name: Dict[int, str], outdir: str, channels: Optional[List[int]] = None) -> List[str]:
    daily = aggregate_daily(event_tensor)  # (days, C)
    days, C = daily.shape
    outputs: List[str] = []

    selected_channels = channels if channels is not None else list(range(C))

    for c in selected_channels:
        series = daily[:, c]
        if np.count_nonzero(series) == 0:
            continue
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(np.arange(days), series, color="tab:blue")
        ax.set_title(
            f"Daily aggregate (sum over regions) - {channel_index_to_name.get(c, f'channel_{c}')} [c={c}]")
        ax.set_xlabel("Day index")
        ax.set_ylabel("Daily sum")
        fig.tight_layout()
        out = os.path.join(outdir, f"temporal_aggregates_channel_{c}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        outputs.append(out)
    return outputs


def find_most_active_day(event_tensor: np.ndarray, channel: int) -> int:
    # compute per-day activity sum for a specific channel, summed across regions and hours
    N_l, T, C = event_tensor.shape
    hours_per_day = 24
    days = T // hours_per_day
    trimmed = event_tensor[:, :days * hours_per_day, channel]
    reshaped = trimmed.reshape(N_l, days, hours_per_day)
    daily_activity = reshaped.sum(axis=(0, 2))  # shape: (days,)
    return int(np.argmax(daily_activity))


def plot_heatmap_most_active_day(event_tensor: np.ndarray, channel: int, channel_name: str, outdir: str) -> Optional[str]:
    N_l, T, C = event_tensor.shape
    hours_per_day = 24
    days = T // hours_per_day
    if days == 0:
        return None
    day_idx = find_most_active_day(event_tensor, channel)
    start = day_idx * hours_per_day
    end = start + hours_per_day
    # slice to regions x 24
    slice_ = event_tensor[:, start:end, channel]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slice_, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(
        f"Most active day heatmap - {channel_name} [c={channel}] - day {day_idx}")
    ax.set_xlabel("Hour (0-23)")
    ax.set_ylabel("Region index (0..N_l-1)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Value")
    fig.tight_layout()
    out = os.path.join(
        outdir, f"heatmap_most_active_day_channel_{channel}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def parse_channel_list(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    indices: List[int] = []
    for p in parts:
        if p.isdigit():
            indices.append(int(p))
    return indices or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize event tensor from NPZ")
    parser.add_argument("--npz", type=str, required=True,
                        help="Path to hk_events_2019.npz")
    parser.add_argument(
        "--outdir", type=str, default="outputs/event_visualizations", help="Directory to save plots")
    parser.add_argument("--channels", type=str, default=None,
                        help="Comma-separated channel indices to visualize (e.g., '6,7,8')")
    parser.add_argument("--save_csv", action="store_true",
                        help="Also save CSV summaries for daily aggregates")
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    event_tensor, dims, ch_idx_to_name, _ = load_npz_with_metadata(args.npz)

    stats = compute_channel_stats(event_tensor)
    sparsity_plot = plot_channel_sparsity(stats, ch_idx_to_name, args.outdir)

    channels = parse_channel_list(args.channels)
    temporal_plots = plot_temporal_aggregates_daily(
        event_tensor, ch_idx_to_name, args.outdir, channels)

    heatmap_plots: List[str] = []
    selected_channels = channels if channels is not None else list(
        range(event_tensor.shape[2]))
    for c in selected_channels:
        if stats[c]["nonzero_count"] == 0:
            continue
        out = plot_heatmap_most_active_day(
            event_tensor, c, ch_idx_to_name.get(c, f"channel_{c}"), args.outdir)
        if out:
            heatmap_plots.append(out)

    if args.save_csv:
        daily = aggregate_daily(event_tensor)  # (days, C)
        days, C = daily.shape
        # Save per-channel daily aggregates
        for c in selected_channels:
            path_csv = os.path.join(
                args.outdir, f"daily_aggregate_channel_{c}.csv")
            np.savetxt(path_csv, daily[:, c], delimiter=",",
                       header="daily_sum", comments="")

    print("Saved:")
    print("  -", sparsity_plot)
    for p in temporal_plots:
        print("  -", p)
    for p in heatmap_plots:
        print("  -", p)


if __name__ == "__main__":
    main()

