import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

from .temporal_processing import TemporalProcessor
from .spatial_mapping import SpatialMapper

logger = logging.getLogger(__name__)

class ConcertProcessor:
    """
    Processes concert events into spatio-temporal tensors.
    Handles venue-based events with attendance and capacity features.
    """
    
    def __init__(self, spatial_mapper: SpatialMapper, temporal_processor: TemporalProcessor):
        """
        Initialize concert processor.
        
        Args:
            spatial_mapper: SpatialMapper instance for region mapping
            temporal_processor: TemporalProcessor for time indexing
        """
        self.spatial_mapper = spatial_mapper
        self.temporal_processor = temporal_processor
        
        # Define venue capacity estimates (based on common Hong Kong venues)
        self.venue_capacities = {
            'hong kong coliseum': 12500,
            'asia world expo': 14000,
            'hong kong stadium': 40000,
            'queen elizabeth stadium': 3500,
            'macpherson stadium': 3750,
            'civic centre': 2000,
            'cultural centre': 2000,
            'city hall': 1500,
            'default': 1000  # Default capacity for unknown venues
        }
    
    def load_concert_events(self, events_file: str) -> pd.DataFrame:
        """
        Load concert events from CSV file.
        
        Args:
            events_file: Path to concert events CSV file
            
        Returns:
            DataFrame with concert events
        """
        try:
            df = pd.read_csv(events_file)
            
            # Clean and standardize column names
            df.columns = df.columns.str.strip().str.lower()
            
            # Ensure required columns exist
            required_cols = ['date', 'venue', 'longitude', 'latitude']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return pd.DataFrame()
            
            logger.info(f"Loaded {len(df)} concert events")
            return df
            
        except Exception as e:
            logger.error(f"Error loading concert events: {e}")
            return pd.DataFrame()
    
    def estimate_venue_capacity(self, venue_name: str) -> int:
        """
        Estimate venue capacity based on venue name.
        
        Args:
            venue_name: Name of the venue
            
        Returns:
            Estimated capacity
        """
        venue_key = venue_name.lower().strip()
        
        # Try exact match first
        if venue_key in self.venue_capacities:
            return self.venue_capacities[venue_key]
        
        # Try partial matching
        for known_venue, capacity in self.venue_capacities.items():
            if known_venue in venue_key or any(word in venue_key for word in known_venue.split()):
                return capacity
        
        # Check for common venue types
        if 'coliseum' in venue_key or 'arena' in venue_key:
            return 10000
        elif 'stadium' in venue_key:
            return 20000
        elif 'hall' in venue_key or 'centre' in venue_key:
            return 2000
        elif 'theatre' in venue_key:
            return 1500
        
        return self.venue_capacities['default']
    
    def calculate_concert_intensity(self, venue_name: str, event_category: str = '') -> float:
        """
        Calculate concert intensity based on venue capacity and event type.
        
        Args:
            venue_name: Name of the venue
            event_category: Category of the event
            
        Returns:
            Intensity score (0.0 to 1.0)
        """
        base_capacity = self.estimate_venue_capacity(venue_name)
        
        # Normalize capacity to intensity (max capacity = 40,000)
        capacity_intensity = min(base_capacity / 40000.0, 1.0)
        
        # Adjust based on event category
        category_multiplier = 1.0
        if event_category:
            category_lower = event_category.lower()
            if 'pop' in category_lower or 'concert' in category_lower:
                category_multiplier = 1.0
            elif 'classical' in category_lower or 'opera' in category_lower:
                category_multiplier = 0.8
            elif 'festival' in category_lower:
                category_multiplier = 1.2
            elif 'comedy' in category_lower or 'talk' in category_lower:
                category_multiplier = 0.6
        
        return min(capacity_intensity * category_multiplier, 1.0)
    
    def calculate_attendance_estimate(self, venue_name: str, event_category: str = '') -> int:
        """
        Estimate event attendance based on venue and type.
        
        Args:
            venue_name: Name of the venue
            event_category: Category of the event
            
        Returns:
            Estimated attendance
        """
        capacity = self.estimate_venue_capacity(venue_name)
        
        # Assume different occupancy rates based on event type
        occupancy_rate = 0.85  # Default 85% occupancy
        
        if event_category:
            category_lower = event_category.lower()
            if 'pop' in category_lower or 'concert' in category_lower:
                occupancy_rate = 0.95  # High demand
            elif 'classical' in category_lower:
                occupancy_rate = 0.75
            elif 'comedy' in category_lower:
                occupancy_rate = 0.80
        
        return int(capacity * occupancy_rate)
    
    def process_events(self, events_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Process concert events into spatio-temporal tensors.
        
        Args:
            events_df: DataFrame with concert events
            
        Returns:
            Dictionary with event tensors and metadata
        """
        num_regions = self.spatial_mapper.region_count
        num_timesteps = self.temporal_processor.get_total_time_steps()
        
        # Initialize tensors
        # concert_presence: binary indicator (0/1)
        concert_presence = np.zeros((num_regions, num_timesteps), dtype=np.float32)
        
        # concert_intensity: intensity based on venue capacity and type
        concert_intensity = np.zeros((num_regions, num_timesteps), dtype=np.float32)
        
        # attendance_level: normalized attendance estimate
        attendance_level = np.zeros((num_regions, num_timesteps), dtype=np.float32)
        
        processed_count = 0
        
        for idx, event in events_df.iterrows():
            try:
                # Parse date and time
                date_str = str(event['date'])
                time_str = event.get('time', '19:30:00')  # Default to 7:30 PM
                
                if pd.isna(time_str) or time_str == 'nan':
                    time_str = '19:30:00'
                
                timestamp_str = f"{date_str} {time_str}"
                time_idx = self.temporal_processor.timestamp_to_time_index(timestamp_str)
                
                if time_idx is None:
                    logger.warning(f"Could not parse timestamp: {timestamp_str}")
                    continue
                
                # Map coordinates to region
                try:
                    lat = float(event['latitude'])
                    lon = float(event['longitude'])
                except (ValueError, TypeError):
                    logger.warning(f"Invalid coordinates for event {idx}")
                    continue
                
                region_idx = self.spatial_mapper.map_coordinates_to_region(lat, lon)
                
                if region_idx is None:
                    logger.warning(f"Could not map coordinates ({lat}, {lon}) to region")
                    continue
                
                # Extract event information
                venue_name = str(event.get('venue', ''))
                event_category = str(event.get('category', ''))
                
                # Calculate event features
                intensity = self.calculate_concert_intensity(venue_name, event_category)
                attendance = self.calculate_attendance_estimate(venue_name, event_category)
                
                # Normalize attendance to [0, 1] range (max = 40,000)
                attendance_normalized = min(attendance / 40000.0, 1.0)
                
                # Set tensor values
                concert_presence[region_idx, time_idx] = 1.0
                
                # Use maximum intensity if multiple concerts in same region/time
                concert_intensity[region_idx, time_idx] = max(
                    concert_intensity[region_idx, time_idx], 
                    intensity
                )
                
                attendance_level[region_idx, time_idx] = max(
                    attendance_level[region_idx, time_idx], 
                    attendance_normalized
                )
                
                # Handle multi-hour events (concerts typically last 2-3 hours)
                for hour_offset in range(1, 4):  # Additional 3 hours
                    extended_time_idx = time_idx + hour_offset
                    if extended_time_idx < num_timesteps:
                        concert_presence[region_idx, extended_time_idx] = 1.0
                        concert_intensity[region_idx, extended_time_idx] = max(
                            concert_intensity[region_idx, extended_time_idx], 
                            intensity * 0.8  # Slightly reduced intensity for extended hours
                        )
                        attendance_level[region_idx, extended_time_idx] = max(
                            attendance_level[region_idx, extended_time_idx], 
                            attendance_normalized * 0.8
                        )
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing concert event {idx}: {e}")
                continue
        
        logger.info(f"Processed {processed_count} concert events")
        
        return {
            'concert_presence': concert_presence,
            'concert_intensity': concert_intensity,
            'attendance_level': attendance_level,
            'metadata': {
                'event_type': 'concert',
                'num_events': processed_count,
                'regions': num_regions,
                'timesteps': num_timesteps,
                'avg_capacity': np.mean(list(self.venue_capacities.values()))
            }
        }
    
    def get_event_statistics(self, tensors: Dict[str, np.ndarray]) -> Dict:
        """
        Calculate statistics for concert events.
        
        Args:
            tensors: Event tensors from process_events
            
        Returns:
            Dictionary with event statistics
        """
        concert_presence = tensors['concert_presence']
        concert_intensity = tensors['concert_intensity']
        attendance_level = tensors['attendance_level']
        
        stats = {
            'total_concert_hours': int(np.sum(concert_presence)),
            'active_regions': int(np.sum(np.any(concert_presence > 0, axis=1))),
            'max_concurrent_concerts': int(np.max(np.sum(concert_presence, axis=0))),
            'avg_intensity': float(np.mean(concert_intensity[concert_intensity > 0])),
            'avg_attendance_level': float(np.mean(attendance_level[attendance_level > 0])),
            'peak_activity_hour': int(np.argmax(np.sum(concert_presence, axis=0))),
            'intensity_distribution': {
                'low_intensity': int(np.sum((concert_intensity > 0) & (concert_intensity <= 0.3))),
                'medium_intensity': int(np.sum((concert_intensity > 0.3) & (concert_intensity <= 0.7))),
                'high_intensity': int(np.sum(concert_intensity > 0.7))
            }
        }
        
        return stats