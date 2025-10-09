"""
Event Processing System for Spatio-Temporal Event Tensors

This module provides a comprehensive system for processing three types of events
into spatio-temporal tensors aligned with hourly time granularity and MTR station-based
Voronoi spatial regions:

1. Service Outages: MTR station disruptions with weighted spatial impact
2. Concert Events: Venue-based events with intensity and attendance features
3. Weather Events: Territory-wide extreme weather warnings (excludes hot weather)

The system generates event tensors E ∈ R^{N_l × T × d_event} where:
- N_l: Number of spatial regions (MTR Voronoi regions)
- T: Number of time steps (hourly for entire year)
- d_event: Number of event feature channels (9 total)

Usage Example:
    from events.tensor_generator import create_event_tensors
    
    # Generate event tensors for 2019
    data_dir = 'data/data_HK'
    output_path = 'events/hk_events_2019.npz'
    tensors = create_event_tensors(data_dir, output_path, year=2019)
    
    # Access the main event tensor
    event_tensor = tensors['event_tensor']  # Shape: (N_l, T, 9)
    
Channel Structure:
    0: Service outage binary indicator (0/1)
    1: Service impact with spatial weights (0.0-1.0)
    2: Disruption intensity by type (0.0-1.0)
    3: Concert presence binary indicator (0/1)
    4: Concert intensity by venue/type (0.0-1.0)  
    5: Concert attendance level normalized (0.0-1.0)
    6: Weather intensity scalar (0.0-1.0)
    7: Weather presence binary indicator (0/1)
    8: Weather spatial coverage factor (0.0-1.0)

Data Requirements:
    - data/data_HK/HK_regions/mtr_unique_stations.csv: MTR station information
    - data/data_HK/HK_regions/mtr_voronoi_final.geojson: Spatial boundaries
    - data/data_HK/Events/mtr_events.txt: MTR service disruption events
    - data/data_HK/Events/Hong_Kong_Concerts_2019_Final_processed.csv: Concert events
    - data/data_HK/Events/weather_warnings_2019_processed.csv: Weather warnings

The system is designed for potential cleanup and reuse in other urban contexts
with appropriate data adaptations.
"""

from .spatial_mapping import SpatialMapper
from .temporal_processing import TemporalProcessor
from .service_outage import ServiceOutageProcessor
from .concert_events import ConcertProcessor
from .weather_events import WeatherProcessor
from .tensor_generator import EventTensorGenerator, create_event_tensors

__version__ = "1.0.0"
__author__ = "HKU Smart Mobility Lab"

__all__ = [
    'SpatialMapper',
    'TemporalProcessor', 
    'ServiceOutageProcessor',
    'ConcertProcessor',
    'WeatherProcessor',
    'EventTensorGenerator',
    'create_event_tensors'
]