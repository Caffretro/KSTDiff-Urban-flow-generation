import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import re
from datetime import datetime

from .temporal_processing import TemporalProcessor
from .spatial_mapping import SpatialMapper

logger = logging.getLogger(__name__)

class ServiceOutageProcessor:
    """
    Processes MTR service outage events into spatio-temporal tensors.
    Handles station-specific disruptions with weighted impact zones.
    """
    
    def __init__(self, spatial_mapper: SpatialMapper, temporal_processor: TemporalProcessor):
        """
        Initialize service outage processor.
        
        Args:
            spatial_mapper: SpatialMapper instance for region mapping
            temporal_processor: TemporalProcessor for time indexing
        """
        self.spatial_mapper = spatial_mapper
        self.temporal_processor = temporal_processor
        
        # Define impact weights for different disruption types
        self.disruption_weights = {
            'closed': 1.0,          # Station completely closed
            'skipped': 0.8,         # Station skipped by trains
            'partial_closed': 0.6,  # Partial closure
            'service_suspended': 0.9,  # Service suspended
            'delayed': 0.3,         # Service delayed
            'limited': 0.4,         # Limited service
            'default': 0.5          # Default weight
        }
    
    def load_mtr_events(self, events_file: str) -> pd.DataFrame:
        """
        Load MTR service outage events from file.
        
        Args:
            events_file: Path to MTR events file
            
        Returns:
            DataFrame with processed events
        """
        events = []
        
        try:
            with open(events_file, 'r', encoding='utf-8') as f:
                current_event = {}
                
                for line in f:
                    line = line.strip()
                    
                    if not line:
                        if current_event:
                            events.append(current_event)
                            current_event = {}
                        continue
                    
                    if line.startswith('Date:'):
                        current_event['date'] = line.replace('Date:', '').strip()
                    elif line.startswith('Time:'):
                        current_event['time'] = line.replace('Time:', '').strip()
                    elif line.startswith('Station:'):
                        current_event['station'] = line.replace('Station:', '').strip()
                    elif line.startswith('Action:'):
                        current_event['action'] = line.replace('Action:', '').strip()
                    elif line.startswith('Note:'):
                        current_event['note'] = line.replace('Note:', '').strip()
                
                # Add last event if exists
                if current_event:
                    events.append(current_event)
        
        except Exception as e:
            logger.error(f"Error loading MTR events: {e}")
            return pd.DataFrame()
        
        df = pd.DataFrame(events)
        logger.info(f"Loaded {len(df)} MTR service events")
        return df
    
    def extract_disruption_type(self, action: str, note: str = "") -> str:
        """
        Extract disruption type from action and note text.
        
        Args:
            action: Action description
            note: Additional note
            
        Returns:
            Disruption type key
        """
        text = f"{action} {note}".lower()
        
        if 'closed' in text and 'partial' in text:
            return 'partial_closed'
        elif 'closed' in text:
            return 'closed'
        elif 'skipped' in text or 'skip' in text:
            return 'skipped'
        elif 'suspended' in text:
            return 'service_suspended'
        elif 'delayed' in text or 'delay' in text:
            return 'delayed'
        elif 'limited' in text:
            return 'limited'
        else:
            return 'default'
    
    def get_disruption_weight(self, disruption_type: str) -> float:
        """Get weight for disruption type."""
        return self.disruption_weights.get(disruption_type, self.disruption_weights['default'])
    
    def process_events(self, events_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Process MTR service outage events into spatio-temporal tensors.
        
        Args:
            events_df: DataFrame with MTR events
            
        Returns:
            Dictionary with event tensors and metadata
        """
        num_regions = self.spatial_mapper.region_count
        num_timesteps = self.temporal_processor.get_total_time_steps()
        
        # Initialize tensors
        # service_outage: binary indicator (0/1)
        service_outage = np.zeros((num_regions, num_timesteps), dtype=np.float32)
        
        # service_impact: weighted impact considering affected zones
        service_impact = np.zeros((num_regions, num_timesteps), dtype=np.float32)
        
        # disruption_intensity: intensity based on disruption type
        disruption_intensity = np.zeros((num_regions, num_timesteps), dtype=np.float32)
        
        processed_count = 0
        
        for idx, event in events_df.iterrows():
            try:
                # Parse timestamp
                timestamp_str = f"{event['date']} {event['time']}"
                time_idx = self.temporal_processor.timestamp_to_time_index(timestamp_str)
                
                if time_idx is None:
                    logger.warning(f"Could not parse timestamp: {timestamp_str}")
                    continue
                
                # Map station to region
                station_name = event['station']
                region_idx = self.spatial_mapper.map_station_to_region(station_name)
                
                if region_idx is None:
                    logger.warning(f"Could not map station: {station_name}")
                    continue
                
                # Extract disruption information
                action = event.get('action', '')
                note = event.get('note', '')
                disruption_type = self.extract_disruption_type(action, note)
                disruption_weight = self.get_disruption_weight(disruption_type)
                
                # Set binary outage indicator
                service_outage[region_idx, time_idx] = 1.0
                
                # Set disruption intensity
                disruption_intensity[region_idx, time_idx] = disruption_weight
                
                # Calculate impact weights for surrounding regions
                impact_weights = self.spatial_mapper.get_region_impact_weights(region_idx)
                
                for affected_region, weight in impact_weights.items():
                    impact_value = disruption_weight * weight
                    service_impact[affected_region, time_idx] = max(
                        service_impact[affected_region, time_idx], 
                        impact_value
                    )
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing event {idx}: {e}")
                continue
        
        logger.info(f"Processed {processed_count} MTR service outage events")
        
        return {
            'service_outage': service_outage,
            'service_impact': service_impact,
            'disruption_intensity': disruption_intensity,
            'metadata': {
                'event_type': 'service_outage',
                'num_events': processed_count,
                'regions': num_regions,
                'timesteps': num_timesteps,
                'disruption_types': list(self.disruption_weights.keys())
            }
        }
    
    def get_event_statistics(self, tensors: Dict[str, np.ndarray]) -> Dict:
        """
        Calculate statistics for service outage events.
        
        Args:
            tensors: Event tensors from process_events
            
        Returns:
            Dictionary with event statistics
        """
        service_outage = tensors['service_outage']
        service_impact = tensors['service_impact']
        disruption_intensity = tensors['disruption_intensity']
        
        stats = {
            'total_outage_hours': int(np.sum(service_outage)),
            'affected_regions': int(np.sum(np.any(service_outage > 0, axis=1))),
            'max_concurrent_outages': int(np.max(np.sum(service_outage, axis=0))),
            'avg_disruption_intensity': float(np.mean(disruption_intensity[disruption_intensity > 0])),
            'peak_impact_hour': int(np.argmax(np.sum(service_impact, axis=0))),
            'impact_coverage': {
                'low_impact': int(np.sum((service_impact > 0) & (service_impact <= 0.3))),
                'medium_impact': int(np.sum((service_impact > 0.3) & (service_impact <= 0.7))),
                'high_impact': int(np.sum(service_impact > 0.7))
            }
        }
        
        return stats