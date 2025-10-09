import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import re
from datetime import datetime

from .temporal_processing import TemporalProcessor
from .spatial_mapping import SpatialMapper

logger = logging.getLogger(__name__)

class WeatherProcessor:
    """
    Processes weather warning events into spatio-temporal tensors.
    Handles extreme weather with scalar intensity encoding (excludes hot weather).
    """
    
    def __init__(self, spatial_mapper: SpatialMapper, temporal_processor: TemporalProcessor):
        """
        Initialize weather processor.
        
        Args:
            spatial_mapper: SpatialMapper instance for region mapping
            temporal_processor: TemporalProcessor for time indexing
        """
        self.spatial_mapper = spatial_mapper
        self.temporal_processor = temporal_processor
        
        # Define intensity weights for different weather warning types
        # Excludes "Very Hot" weather as specified
        self.weather_intensities = {
            # Tropical Cyclones (highest intensity)
            'tropical cyclone': 1.0,
            'typhoon': 1.0,
            'hurricane': 1.0,
            
            # Rainstorm signals (high intensity)
            'red': 0.9,      # Red rainstorm
            'amber': 0.7,    # Amber rainstorm
            'yellow': 0.5,   # Yellow rainstorm (if mentioned)
            
            # Thunderstorm warnings (medium-high intensity)
            'thunderstorm': 0.6,
            
            # Wind warnings (medium intensity)
            'strong monsoon': 0.5,
            'gale': 0.7,
            
            # Cold weather (medium-low intensity)
            'cold weather': 0.4,
            'frost': 0.6,
            
            # Default for unspecified warnings
            'default': 0.3
        }
    
    def load_weather_events(self, events_file: str) -> pd.DataFrame:
        """
        Load weather warning events from CSV file.
        
        Args:
            events_file: Path to weather events CSV file
            
        Returns:
            DataFrame with weather events
        """
        try:
            df = pd.read_csv(events_file)
            
            # Clean column names
            df.columns = df.columns.str.strip().str.lower()
            
            # Filter out "Very Hot" weather warnings
            if 'warning_type' in df.columns:
                initial_count = len(df)
                df = df[~df['warning_type'].str.contains('Very Hot', case=False, na=False)]
                filtered_count = initial_count - len(df)
                logger.info(f"Filtered out {filtered_count} 'Very Hot' weather warnings")
            
            logger.info(f"Loaded {len(df)} weather warning events")
            return df
            
        except Exception as e:
            logger.error(f"Error loading weather events: {e}")
            return pd.DataFrame()
    
    def extract_weather_intensity(self, warning_type: str, warning_title: str = '', 
                                warning_content: str = '') -> float:
        """
        Extract weather intensity from warning information.
        
        Args:
            warning_type: Type of weather warning
            warning_title: Title of the warning
            warning_content: Content/description of the warning
            
        Returns:
            Intensity score (0.0 to 1.0)
        """
        # Combine all text for analysis
        text = f"{warning_type} {warning_title} {warning_content}".lower()
        
        # Skip hot weather warnings
        if 'very hot' in text or 'hot weather' in text:
            return 0.0
        
        # Check for tropical cyclone signals (highest priority)
        cyclone_signals = re.findall(r'signal.*no\.?\s*(\d+)', text)
        if cyclone_signals:
            signal_num = int(cyclone_signals[0])
            if signal_num >= 8:
                return 1.0  # Severe typhoon
            elif signal_num >= 3:
                return 0.8  # Strong wind signal
            else:
                return 0.4  # Standby signal
        
        # Check for rainstorm warnings
        if 'red rainstorm' in text or 'red rain' in text:
            return 0.9
        elif 'amber rainstorm' in text or 'amber rain' in text:
            return 0.7
        elif 'yellow rainstorm' in text or 'yellow rain' in text:
            return 0.5
        
        # Check for specific wind speeds or conditions
        wind_speeds = re.findall(r'(\d+)\s*kilometres?\s*per\s*hour', text)
        if wind_speeds:
            max_wind = max(int(speed) for speed in wind_speeds)
            if max_wind >= 100:
                return 0.9  # Hurricane force
            elif max_wind >= 70:
                return 0.7  # Gale force
            elif max_wind >= 40:
                return 0.5  # Strong wind
        
        # Check for specific warning types
        for weather_type, intensity in self.weather_intensities.items():
            if weather_type in text:
                return intensity
        
        # Default intensity for unrecognized warnings
        return self.weather_intensities['default']
    
    def determine_spatial_coverage(self, warning_content: str) -> float:
        """
        Determine spatial coverage of weather event.
        
        Args:
            warning_content: Content of weather warning
            
        Returns:
            Coverage factor (0.0 to 1.0) - 1.0 means territory-wide
        """
        content_lower = warning_content.lower()
        
        # Territory-wide events
        if any(phrase in content_lower for phrase in [
            'hong kong', 'territory', 'generally over hong kong', 'across hong kong'
        ]):
            return 1.0
        
        # Regional events
        elif any(phrase in content_lower for phrase in [
            'new territories', 'kowloon', 'hong kong island', 'lantau'
        ]):
            return 0.7
        
        # Local events
        elif any(phrase in content_lower for phrase in [
            'isolated', 'few', 'local', 'specific areas'
        ]):
            return 0.3
        
        # Default to moderate coverage
        return 0.6
    
    def process_events(self, events_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Process weather warning events into spatio-temporal tensors.
        
        Args:
            events_df: DataFrame with weather events
            
        Returns:
            Dictionary with event tensors and metadata
        """
        num_regions = self.spatial_mapper.region_count
        num_timesteps = self.temporal_processor.get_total_time_steps()
        
        # Initialize tensors
        # weather_intensity: scalar intensity of weather warnings
        weather_intensity = np.zeros((num_regions, num_timesteps), dtype=np.float32)
        
        # weather_presence: binary indicator of any weather warning
        weather_presence = np.zeros((num_regions, num_timesteps), dtype=np.float32)
        
        # weather_coverage: spatial coverage factor
        weather_coverage = np.zeros((num_regions, num_timesteps), dtype=np.float32)
        
        processed_count = 0
        
        for idx, event in events_df.iterrows():
            try:
                # Parse time range
                start_time = str(event['start_time'])
                end_time = str(event.get('end_time', ''))
                
                if pd.isna(end_time) or end_time == 'nan':
                    # Default duration: 6 hours for weather warnings
                    time_indices = self.temporal_processor.get_time_range(
                        start_time, duration_hours=6
                    )
                else:
                    time_indices = self.temporal_processor.get_time_range(
                        start_time, end_time
                    )
                
                if not time_indices:
                    logger.warning(f"Could not parse time range: {start_time} to {end_time}")
                    continue
                
                # Extract warning information
                warning_type = str(event.get('warning_type', ''))
                warning_title = str(event.get('warning_title', ''))
                warning_content = str(event.get('warning_content', ''))
                
                # Skip hot weather warnings
                if 'very hot' in warning_type.lower():
                    continue
                
                # Calculate weather intensity
                intensity = self.extract_weather_intensity(
                    warning_type, warning_title, warning_content
                )
                
                if intensity == 0.0:  # Skip if it's hot weather or invalid
                    continue
                
                # Determine spatial coverage
                coverage_factor = self.determine_spatial_coverage(warning_content)
                
                # Apply to all regions (weather affects entire territory)
                all_regions = self.spatial_mapper.get_all_regions()
                
                for region_idx in all_regions:
                    for time_idx in time_indices:
                        if time_idx < num_timesteps:
                            # Use maximum intensity if multiple warnings overlap
                            weather_intensity[region_idx, time_idx] = max(
                                weather_intensity[region_idx, time_idx],
                                intensity * coverage_factor
                            )
                            
                            weather_presence[region_idx, time_idx] = 1.0
                            
                            weather_coverage[region_idx, time_idx] = max(
                                weather_coverage[region_idx, time_idx],
                                coverage_factor
                            )
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing weather event {idx}: {e}")
                continue
        
        logger.info(f"Processed {processed_count} weather warning events")
        
        return {
            'weather_intensity': weather_intensity,
            'weather_presence': weather_presence,
            'weather_coverage': weather_coverage,
            'metadata': {
                'event_type': 'weather',
                'num_events': processed_count,
                'regions': num_regions,
                'timesteps': num_timesteps,
                'intensity_types': list(self.weather_intensities.keys())
            }
        }
    
    def get_event_statistics(self, tensors: Dict[str, np.ndarray]) -> Dict:
        """
        Calculate statistics for weather events.
        
        Args:
            tensors: Event tensors from process_events
            
        Returns:
            Dictionary with event statistics
        """
        weather_intensity = tensors['weather_intensity']
        weather_presence = tensors['weather_presence']
        weather_coverage = tensors['weather_coverage']
        
        stats = {
            'total_warning_hours': int(np.sum(weather_presence)),
            'avg_intensity': float(np.mean(weather_intensity[weather_intensity > 0])),
            'max_intensity': float(np.max(weather_intensity)),
            'avg_coverage': float(np.mean(weather_coverage[weather_coverage > 0])),
            'peak_weather_hour': int(np.argmax(np.sum(weather_intensity, axis=0))),
            'intensity_distribution': {
                'low_intensity': int(np.sum((weather_intensity > 0) & (weather_intensity <= 0.3))),
                'medium_intensity': int(np.sum((weather_intensity > 0.3) & (weather_intensity <= 0.7))),
                'high_intensity': int(np.sum(weather_intensity > 0.7))
            },
            'coverage_distribution': {
                'local_coverage': int(np.sum((weather_coverage > 0) & (weather_coverage <= 0.4))),
                'regional_coverage': int(np.sum((weather_coverage > 0.4) & (weather_coverage <= 0.8))),
                'territory_wide': int(np.sum(weather_coverage > 0.8))
            }
        }
        
        return stats