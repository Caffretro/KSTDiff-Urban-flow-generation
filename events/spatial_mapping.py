import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpatialMapper:
    """
    Maps events to Voronoi regions based on MTR stations.
    Handles spatial mapping for station outages, concerts, and weather events.
    """
    
    def __init__(self, stations_file: str, voronoi_file: str):
        """
        Initialize spatial mapper with station and Voronoi data.
        
        Args:
            stations_file: Path to MTR stations CSV file
            voronoi_file: Path to Voronoi regions GeoJSON file
        """
        self.stations_df = pd.read_csv(stations_file)
        self.voronoi_gdf = gpd.read_file(voronoi_file)
        
        # Create station name to region mapping
        self.station_to_region = self._create_station_mapping()
        self.region_count = len(self.voronoi_gdf)
        
        logger.info(f"Initialized spatial mapper with {self.region_count} regions")
    
    def _create_station_mapping(self) -> Dict[str, int]:
        """Create mapping from station names to region indices."""
        mapping = {}
        
        # Map station codes/names to region indices
        for idx, row in self.stations_df.iterrows():
            station_name = row['English Name'].lower()
            station_code = row['Station Code']
            
            # Find corresponding Voronoi region
            for region_idx, region_row in self.voronoi_gdf.iterrows():
                # Match by station name or code in the Voronoi data
                if 'Station name' in region_row and region_row['Station name'].lower() == station_name:
                    mapping[station_name] = region_idx
                    mapping[station_code] = region_idx
                    break
                elif 'Station code' in region_row and region_row['Station code'] == station_code:
                    mapping[station_name] = region_idx
                    mapping[station_code] = region_idx
                    break
            
            # If no exact match found, use index directly
            if station_name not in mapping:
                mapping[station_name] = idx
                mapping[station_code] = idx
        
        return mapping
    
    def map_station_to_region(self, station_name: str) -> Optional[int]:
        """
        Map a station name to its Voronoi region index.
        
        Args:
            station_name: Name or code of the station
            
        Returns:
            Region index or None if not found
        """
        # Try exact match first
        station_key = station_name.lower().strip()
        if station_key in self.station_to_region:
            return self.station_to_region[station_key]
        
        # Try fuzzy matching for station names
        for mapped_name, region_idx in self.station_to_region.items():
            if station_key in mapped_name or mapped_name in station_key:
                return region_idx
        
        logger.warning(f"Station '{station_name}' not found in mapping")
        return None
    
    def map_coordinates_to_region(self, lat: float, lon: float) -> Optional[int]:
        """
        Map coordinates to the closest Voronoi region.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Region index or None if mapping fails
        """
        try:
            point = Point(lon, lat)  # Note: Point(x, y) where x=longitude, y=latitude
            
            # Find which Voronoi region contains this point
            for idx, region in self.voronoi_gdf.iterrows():
                if region.geometry.contains(point):
                    return idx
            
            # If no region contains the point, find the closest one
            distances = []
            for idx, region in self.voronoi_gdf.iterrows():
                distance = point.distance(region.geometry)
                distances.append((distance, idx))
            
            if distances:
                closest_region = min(distances, key=lambda x: x[0])[1]
                logger.info(f"Point ({lat}, {lon}) mapped to closest region {closest_region}")
                return closest_region
                
        except Exception as e:
            logger.error(f"Error mapping coordinates ({lat}, {lon}): {e}")
        
        return None
    
    def get_region_impact_weights(self, affected_region: int, buffer_zones: int = 2) -> Dict[int, float]:
        """
        Calculate impact weights for affected region and surrounding areas.
        
        Args:
            affected_region: Index of directly affected region
            buffer_zones: Number of buffer zones to consider
            
        Returns:
            Dictionary mapping region indices to impact weights (0.0 to 1.0)
        """
        weights = {}
        
        if affected_region is None:
            return weights
        
        # Direct impact
        weights[affected_region] = 1.0
        
        try:
            # Get geometry of affected region
            affected_geom = self.voronoi_gdf.iloc[affected_region].geometry
            
            # Calculate impacts for other regions based on distance
            for idx, region in self.voronoi_gdf.iterrows():
                if idx != affected_region:
                    distance = affected_geom.distance(region.geometry)
                    
                    # Apply distance-based decay (inverse exponential)
                    # Regions closer get higher weights
                    if distance == 0:  # Adjacent regions
                        weights[idx] = 0.7
                    else:
                        # Exponential decay based on distance
                        weight = np.exp(-distance * 2)  # Adjust the multiplier as needed
                        if weight > 0.1:  # Only include significant impacts
                            weights[idx] = weight
            
        except Exception as e:
            logger.error(f"Error calculating impact weights for region {affected_region}: {e}")
        
        return weights
    
    def get_all_regions(self) -> List[int]:
        """Get list of all region indices."""
        return list(range(self.region_count))
    
    def get_region_info(self, region_idx: int) -> Dict:
        """Get information about a specific region."""
        if region_idx < len(self.voronoi_gdf):
            region_data = self.voronoi_gdf.iloc[region_idx]
            return {
                'region_id': region_idx,
                'geometry': region_data.geometry,
                'properties': {k: v for k, v in region_data.items() if k != 'geometry'}
            }
        return {}