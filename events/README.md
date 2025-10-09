# Event Processing System

This directory contains a comprehensive system for processing three types of events into spatio-temporal tensors:

## Event Types

1. **Service Outages** (`service_outage.py`)
   - MTR station disruptions with weighted spatial impact
   - Binary indicators (0/1) and impact weights considering buffer zones
   - Disruption intensity based on closure type (closed, skipped, partial, etc.)

2. **Concert Events** (`concert_events.py`) 
   - Venue-based events with intensity and attendance features
   - Binary presence indicators and capacity-based intensity
   - Attendance estimates based on venue size and event type

3. **Weather Events** (`weather_events.py`)
   - Extreme weather warnings with scalar intensity (excludes hot weather)
   - Territory-wide coverage with spatial distribution factors
   - Intensity based on warning type (tropical cyclone, rainstorm, etc.)

## System Components

- **`spatial_mapping.py`**: Maps events to MTR station-based Voronoi regions
- **`temporal_processing.py`**: Handles hourly time alignment and indexing
- **`tensor_generator.py`**: Main orchestrator that combines all event types
- **`example_usage.py`**: Demonstration script showing how to use the system

## Output Format

The system generates event tensors **E ∈ R^{N_l × T × d_event}** where:
- **N_l**: Number of spatial regions (MTR Voronoi regions, ~26)
- **T**: Number of time steps (8760/8784 hours for full year)
- **d_event**: 9 event feature channels

### Channel Structure
```
0: Service outage binary indicator (0/1)
1: Service impact with spatial weights (0.0-1.0) 
2: Disruption intensity by type (0.0-1.0)
3: Concert presence binary indicator (0/1)
4: Concert intensity by venue/type (0.0-1.0)
5: Concert attendance level normalized (0.0-1.0)
6: Weather intensity scalar (0.0-1.0)
7: Weather presence binary indicator (0/1)
8: Weather spatial coverage factor (0.0-1.0)
```

## Usage

```python
from events.tensor_generator import create_event_tensors

# Generate event tensors for 2019
data_dir = 'data/data_HK'
output_path = 'events/hk_events_2019.npz'
tensors = create_event_tensors(data_dir, output_path, year=2019)

# Access the main event tensor
event_tensor = tensors['event_tensor']  # Shape: (N_l, T, 9)
```

## Data Requirements

The system expects the following data files:
- `data/data_HK/HK_regions/mtr_unique_stations.csv`: MTR station information
- `data/data_HK/HK_regions/mtr_voronoi_final.geojson`: Spatial boundaries
- `data/data_HK/Events/mtr_events.txt`: MTR service disruption events
- `data/data_HK/Events/Hong_Kong_Concerts_2019_Final_processed.csv`: Concert events
- `data/data_HK/Events/weather_warnings_2019_processed.csv`: Weather warnings

## Features

- **Modular Design**: Each event type is processed independently and can be used separately
- **Spatial Intelligence**: Sophisticated mapping of events to geographic regions with impact zones
- **Temporal Precision**: Hourly alignment with proper handling of event durations
- **Intensity Encoding**: Context-aware intensity calculation for different event characteristics
- **Statistical Analysis**: Comprehensive statistics and metadata for model validation
- **Error Handling**: Robust parsing and processing with detailed logging

The system is designed for potential cleanup and reuse in other urban contexts with appropriate data adaptations.