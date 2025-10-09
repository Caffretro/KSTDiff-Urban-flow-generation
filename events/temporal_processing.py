import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
import re

logger = logging.getLogger(__name__)

class TemporalProcessor:
    """
    Handles temporal alignment of events to hourly time intervals.
    Creates time indices and manages temporal features for event tensors.
    """
    
    def __init__(self, year: int = 2019, time_format: str = "YYYY-MM-DD-HH"):
        """
        Initialize temporal processor for a specific year.
        
        Args:
            year: Year for temporal processing (default 2019)
            time_format: Time format string for indexing
        """
        self.year = year
        self.time_format = time_format
        
        # Generate hourly time indices for the entire year
        self.time_indices = self._generate_time_indices()
        self.time_to_index = {time_str: idx for idx, time_str in enumerate(self.time_indices)}
        
        logger.info(f"Generated {len(self.time_indices)} hourly time indices for {year}")
    
    def _generate_time_indices(self) -> List[str]:
        """Generate hourly time indices for the entire year."""
        indices = []
        start_date = datetime(self.year, 1, 1, 0, 0, 0)
        
        # Determine if leap year
        if self.year % 4 == 0 and (self.year % 100 != 0 or self.year % 400 == 0):
            total_hours = 366 * 24
        else:
            total_hours = 365 * 24
        
        for hour_offset in range(total_hours):
            current_time = start_date + timedelta(hours=hour_offset)
            time_str = current_time.strftime("%Y-%m-%d-%H")
            indices.append(time_str)
        
        return indices
    
    def parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """
        Parse various timestamp formats into datetime objects.
        
        Args:
            timestamp_str: Timestamp string in various formats
            
        Returns:
            datetime object or None if parsing fails
        """
        # Common timestamp formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y",
            "%Y%m%d",
            "%d-%m-%Y %H:%M:%S",
            "%d-%m-%Y %H:%M",
            "%d-%m-%Y"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str.strip(), fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse timestamp: {timestamp_str}")
        return None
    
    def datetime_to_time_index(self, dt: datetime) -> Optional[int]:
        """
        Convert datetime to time index in the hourly grid.
        
        Args:
            dt: datetime object
            
        Returns:
            Time index or None if out of range
        """
        if dt.year != self.year:
            logger.warning(f"Datetime year {dt.year} doesn't match target year {self.year}")
            return None
        
        time_str = dt.strftime("%Y-%m-%d-%H")
        return self.time_to_index.get(time_str)
    
    def timestamp_to_time_index(self, timestamp_str: str) -> Optional[int]:
        """
        Convert timestamp string to time index.
        
        Args:
            timestamp_str: Timestamp string
            
        Returns:
            Time index or None if parsing/conversion fails
        """
        dt = self.parse_timestamp(timestamp_str)
        if dt is None:
            return None
        return self.datetime_to_time_index(dt)
    
    def get_time_range(self, start_timestamp: str, end_timestamp: Optional[str] = None, 
                      duration_hours: Optional[int] = None) -> List[int]:
        """
        Get list of time indices for an event duration.
        
        Args:
            start_timestamp: Event start time
            end_timestamp: Event end time (optional)
            duration_hours: Duration in hours if end_timestamp not provided
            
        Returns:
            List of time indices covering the event duration
        """
        start_dt = self.parse_timestamp(start_timestamp)
        if start_dt is None:
            return []
        
        # Determine end time
        if end_timestamp:
            end_dt = self.parse_timestamp(end_timestamp)
            if end_dt is None:
                end_dt = start_dt + timedelta(hours=1)  # Default 1 hour
        elif duration_hours:
            end_dt = start_dt + timedelta(hours=duration_hours)
        else:
            end_dt = start_dt + timedelta(hours=1)  # Default 1 hour
        
        # Generate time indices for the duration
        indices = []
        current_dt = start_dt
        
        while current_dt <= end_dt:
            idx = self.datetime_to_time_index(current_dt)
            if idx is not None:
                indices.append(idx)
            current_dt += timedelta(hours=1)
        
        return indices
    
    def extract_date_from_filename(self, filename: str) -> Optional[str]:
        """
        Extract date from filename patterns like 'YYYYMMDD-HHMM-filename.ext'.
        
        Args:
            filename: Filename containing date pattern
            
        Returns:
            Date string or None if not found
        """
        # Pattern for YYYYMMDD-HHMM
        pattern = r'(\d{8})-(\d{4})'
        match = re.search(pattern, filename)
        
        if match:
            date_part = match.group(1)  # YYYYMMDD
            time_part = match.group(2)  # HHMM
            
            # Convert to YYYY-MM-DD HH:MM format
            year = date_part[:4]
            month = date_part[4:6]
            day = date_part[6:8]
            hour = time_part[:2]
            minute = time_part[2:4]
            
            return f"{year}-{month}-{day} {hour}:{minute}:00"
        
        return None
    
    def get_time_features(self, time_indices: List[int]) -> Dict[str, List[float]]:
        """
        Extract temporal features for time indices.
        
        Args:
            time_indices: List of time indices
            
        Returns:
            Dictionary of temporal features
        """
        features = {
            'hour_of_day': [],
            'day_of_week': [],
            'day_of_month': [],
            'month_of_year': [],
            'is_weekend': [],
            'is_holiday': []  # Simplified - could be enhanced with actual holiday calendar
        }
        
        for idx in time_indices:
            if idx < len(self.time_indices):
                time_str = self.time_indices[idx]
                dt = datetime.strptime(time_str, "%Y-%m-%d-%H")
                
                features['hour_of_day'].append(dt.hour)
                features['day_of_week'].append(dt.weekday())
                features['day_of_month'].append(dt.day)
                features['month_of_year'].append(dt.month)
                features['is_weekend'].append(1.0 if dt.weekday() >= 5 else 0.0)
                
                # Simplified holiday detection (New Year, Christmas, etc.)
                is_holiday = 0.0
                if (dt.month == 1 and dt.day == 1) or \
                   (dt.month == 12 and dt.day == 25) or \
                   (dt.month == 10 and dt.day == 1):  # National Day
                    is_holiday = 1.0
                
                features['is_holiday'].append(is_holiday)
        
        return features
    
    def normalize_temporal_features(self, features: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        Normalize temporal features to [0, 1] range.
        
        Args:
            features: Dictionary of temporal features
            
        Returns:
            Normalized features
        """
        normalized = {}
        
        for feature_name, values in features.items():
            if not values:
                normalized[feature_name] = []
                continue
                
            if feature_name == 'hour_of_day':
                # Normalize hours to [0, 1]
                normalized[feature_name] = [v / 23.0 for v in values]
            elif feature_name == 'day_of_week':
                # Normalize weekdays to [0, 1]
                normalized[feature_name] = [v / 6.0 for v in values]
            elif feature_name == 'day_of_month':
                # Normalize days to [0, 1]
                normalized[feature_name] = [(v - 1) / 30.0 for v in values]
            elif feature_name == 'month_of_year':
                # Normalize months to [0, 1]
                normalized[feature_name] = [(v - 1) / 11.0 for v in values]
            else:
                # Keep binary features as is
                normalized[feature_name] = values
        
        return normalized
    
    def get_total_time_steps(self) -> int:
        """Get total number of time steps in the year."""
        return len(self.time_indices)
    
    def time_index_to_string(self, time_idx: int) -> Optional[str]:
        """Convert time index back to time string."""
        if 0 <= time_idx < len(self.time_indices):
            return self.time_indices[time_idx]
        return None