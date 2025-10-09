import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import os

from .spatial_mapping import SpatialMapper
from .temporal_processing import TemporalProcessor
from .service_outage import ServiceOutageProcessor
from .concert_events import ConcertProcessor
from .weather_events import WeatherProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventTensorGenerator:
    """
    Main class for generating spatio-temporal event tensors.
    Combines MTR service outages, concerts, and weather events into unified tensors.
    """
    
    def __init__(self, data_dir: str, year: int = 2019):
        """
        Initialize the event tensor generator.
        
        Args:
            data_dir: Base directory containing data files
            year: Year for processing (default 2019)
        """
        self.data_dir = data_dir
        self.year = year
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"EventTensorGenerator initialized for year {year}")
    
    def _initialize_components(self):
        """Initialize spatial mapper, temporal processor, and event processors."""
        # File paths
        stations_file = os.path.join(self.data_dir, 'HK_regions', 'mtr_unique_stations.csv')
        voronoi_file = os.path.join(self.data_dir, 'HK_regions', 'mtr_voronoi_final.geojson')
        
        # Initialize core components
        self.spatial_mapper = SpatialMapper(stations_file, voronoi_file)
        self.temporal_processor = TemporalProcessor(year=self.year)
        
        # Initialize event processors
        self.service_processor = ServiceOutageProcessor(
            self.spatial_mapper, self.temporal_processor
        )
        self.concert_processor = ConcertProcessor(
            self.spatial_mapper, self.temporal_processor
        )
        self.weather_processor = WeatherProcessor(
            self.spatial_mapper, self.temporal_processor
        )
    
    def generate_event_tensors(self) -> Dict[str, np.ndarray]:
        """
        Generate complete event tensors for all event types.
        
        Returns:
            Dictionary containing event tensors E ∈ R^{N_l × T × d_event}
        """
        logger.info("Starting event tensor generation...")
        
        # File paths for event data
        mtr_events_file = os.path.join(self.data_dir, 'Events', 'mtr_events.txt')
        concert_events_file = os.path.join(self.data_dir, 'Events', 'Hong_Kong_Concerts_2019_Final_processed.csv')
        weather_events_file = os.path.join(self.data_dir, 'Events', 'weather_warnings_2019_processed.csv')
        
        # Process each event type
        service_tensors = self._process_service_outages(mtr_events_file)
        concert_tensors = self._process_concerts(concert_events_file)
        weather_tensors = self._process_weather(weather_events_file)
        
        # Combine into unified event tensor
        combined_tensors = self._combine_tensors(
            service_tensors, concert_tensors, weather_tensors
        )
        
        logger.info("Event tensor generation completed")
        return combined_tensors
    
    def _process_service_outages(self, events_file: str) -> Dict[str, np.ndarray]:
        """Process MTR service outage events."""
        logger.info("Processing MTR service outage events...")
        
        if not os.path.exists(events_file):
            logger.error(f"MTR events file not found: {events_file}")
            return {}
        
        # Load and process events
        events_df = self.service_processor.load_mtr_events(events_file)
        if events_df.empty:
            logger.warning("No MTR events loaded")
            return {}
        
        tensors = self.service_processor.process_events(events_df)
        stats = self.service_processor.get_event_statistics(tensors)
        
        logger.info(f"Service outage stats: {stats}")
        return tensors
    
    def _process_concerts(self, events_file: str) -> Dict[str, np.ndarray]:
        """Process concert events."""
        logger.info("Processing concert events...")
        
        if not os.path.exists(events_file):
            logger.error(f"Concert events file not found: {events_file}")
            return {}
        
        # Load and process events
        events_df = self.concert_processor.load_concert_events(events_file)
        if events_df.empty:
            logger.warning("No concert events loaded")
            return {}
        
        tensors = self.concert_processor.process_events(events_df)
        stats = self.concert_processor.get_event_statistics(tensors)
        
        logger.info(f"Concert stats: {stats}")
        return tensors
    
    def _process_weather(self, events_file: str) -> Dict[str, np.ndarray]:
        """Process weather warning events."""
        logger.info("Processing weather warning events...")
        
        if not os.path.exists(events_file):
            logger.error(f"Weather events file not found: {events_file}")
            return {}
        
        # Load and process events
        events_df = self.weather_processor.load_weather_events(events_file)
        if events_df.empty:
            logger.warning("No weather events loaded")
            return {}
        
        tensors = self.weather_processor.process_events(events_df)
        stats = self.weather_processor.get_event_statistics(tensors)
        
        logger.info(f"Weather stats: {stats}")
        return tensors
    
    def _combine_tensors(self, service_tensors: Dict, concert_tensors: Dict, 
                        weather_tensors: Dict) -> Dict[str, np.ndarray]:
        """
        Combine individual event tensors into unified event tensor.
        
        Args:
            service_tensors: Service outage tensors
            concert_tensors: Concert event tensors
            weather_tensors: Weather event tensors
            
        Returns:
            Combined event tensors E ∈ R^{N_l × T × d_event}
        """
        num_regions = self.spatial_mapper.region_count
        num_timesteps = self.temporal_processor.get_total_time_steps()
        
        # Define event tensor dimensions
        # Service outage: 3 channels (presence, impact, intensity)
        # Concert: 3 channels (presence, intensity, attendance)
        # Weather: 3 channels (intensity, presence, coverage)
        d_event = 9  # Total event dimensions
        
        # Initialize combined tensor
        event_tensor = np.zeros((num_regions, num_timesteps, d_event), dtype=np.float32)
        
        # Channel assignments
        channels = {
            'service_outage': 0,      # Binary service outage
            'service_impact': 1,      # Weighted service impact
            'disruption_intensity': 2, # Disruption intensity
            'concert_presence': 3,    # Binary concert presence
            'concert_intensity': 4,   # Concert intensity
            'attendance_level': 5,    # Concert attendance level
            'weather_intensity': 6,   # Weather intensity
            'weather_presence': 7,    # Weather presence
            'weather_coverage': 8     # Weather coverage
        }
        
        # Fill service outage channels
        if service_tensors:
            if 'service_outage' in service_tensors:
                event_tensor[:, :, channels['service_outage']] = service_tensors['service_outage']
            if 'service_impact' in service_tensors:
                event_tensor[:, :, channels['service_impact']] = service_tensors['service_impact']
            if 'disruption_intensity' in service_tensors:
                event_tensor[:, :, channels['disruption_intensity']] = service_tensors['disruption_intensity']
        
        # Fill concert channels
        if concert_tensors:
            if 'concert_presence' in concert_tensors:
                event_tensor[:, :, channels['concert_presence']] = concert_tensors['concert_presence']
            if 'concert_intensity' in concert_tensors:
                event_tensor[:, :, channels['concert_intensity']] = concert_tensors['concert_intensity']
            if 'attendance_level' in concert_tensors:
                event_tensor[:, :, channels['attendance_level']] = concert_tensors['attendance_level']
        
        # Fill weather channels
        if weather_tensors:
            if 'weather_intensity' in weather_tensors:
                event_tensor[:, :, channels['weather_intensity']] = weather_tensors['weather_intensity']
            if 'weather_presence' in weather_tensors:
                event_tensor[:, :, channels['weather_presence']] = weather_tensors['weather_presence']
            if 'weather_coverage' in weather_tensors:
                event_tensor[:, :, channels['weather_coverage']] = weather_tensors['weather_coverage']
        
        # Create metadata
        metadata = {
            'tensor_shape': event_tensor.shape,
            'dimensions': {
                'N_l': num_regions,
                'T': num_timesteps,
                'd_event': d_event
            },
            'channels': channels,
            'channel_descriptions': {
                0: 'Service outage binary indicator',
                1: 'Service impact with spatial weights',
                2: 'Disruption intensity by type',
                3: 'Concert presence binary indicator',
                4: 'Concert intensity by venue/type',
                5: 'Concert attendance level normalized',
                6: 'Weather intensity scalar',
                7: 'Weather presence binary indicator',
                8: 'Weather spatial coverage factor'
            },
            'year': self.year,
            'temporal_resolution': 'hourly',
            'spatial_resolution': 'MTR Voronoi regions'
        }
        
        # Calculate combined statistics
        combined_stats = self._calculate_combined_statistics(
            event_tensor, service_tensors, concert_tensors, weather_tensors
        )
        
        return {
            'event_tensor': event_tensor,
            'metadata': metadata,
            'statistics': combined_stats,
            'service_tensors': service_tensors,
            'concert_tensors': concert_tensors,
            'weather_tensors': weather_tensors
        }
    
    def _calculate_combined_statistics(self, event_tensor: np.ndarray,
                                     service_tensors: Dict, concert_tensors: Dict,
                                     weather_tensors: Dict) -> Dict:
        """Calculate statistics for the combined event tensor."""
        stats = {
            'tensor_statistics': {
                'total_nonzero_entries': int(np.sum(event_tensor > 0)),
                'sparsity': float(1.0 - np.count_nonzero(event_tensor) / event_tensor.size),
                'max_value': float(np.max(event_tensor)),
                'mean_nonzero': float(np.mean(event_tensor[event_tensor > 0])) if np.any(event_tensor > 0) else 0.0,
                'active_regions': int(np.sum(np.any(event_tensor > 0, axis=(1, 2)))),
                'active_timesteps': int(np.sum(np.any(event_tensor > 0, axis=(0, 2))))
            },
            'channel_statistics': {}
        }
        
        # Per-channel statistics
        for channel_idx in range(event_tensor.shape[2]):
            channel_data = event_tensor[:, :, channel_idx]
            stats['channel_statistics'][channel_idx] = {
                'nonzero_count': int(np.sum(channel_data > 0)),
                'max_value': float(np.max(channel_data)),
                'mean_nonzero': float(np.mean(channel_data[channel_data > 0])) if np.any(channel_data > 0) else 0.0,
                'sparsity': float(1.0 - np.count_nonzero(channel_data) / channel_data.size)
            }
        
        return stats
    
    def save_tensors(self, tensors: Dict, output_path: str):
        """
        Save event tensors to file.
        
        Args:
            tensors: Dictionary containing tensors and metadata
            output_path: Path to save the tensors
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as .npz file
            np.savez_compressed(
                output_path,
                event_tensor=tensors['event_tensor'],
                metadata=json.dumps(tensors['metadata']),
                statistics=json.dumps(tensors['statistics'])
            )
            
            logger.info(f"Event tensors saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving tensors: {e}")
    
    def load_tensors(self, input_path: str) -> Dict:
        """
        Load event tensors from file.
        
        Args:
            input_path: Path to load tensors from
            
        Returns:
            Dictionary containing tensors and metadata
        """
        try:
            data = np.load(input_path, allow_pickle=True)
            
            return {
                'event_tensor': data['event_tensor'],
                'metadata': json.loads(str(data['metadata'])),
                'statistics': json.loads(str(data['statistics']))
            }
            
        except Exception as e:
            logger.error(f"Error loading tensors: {e}")
            return {}

def create_event_tensors(data_dir: str, output_path: str, year: int = 2019) -> Dict[str, np.ndarray]:
    """
    Convenience function to create event tensors.
    
    Args:
        data_dir: Directory containing event data files
        output_path: Path to save the generated tensors
        year: Year for processing
        
    Returns:
        Dictionary containing generated tensors
    """
    generator = EventTensorGenerator(data_dir, year)
    tensors = generator.generate_event_tensors()
    
    if output_path:
        generator.save_tensors(tensors, output_path)
    
    return tensors