#!/usr/bin/env python3
"""
Example script demonstrating event tensor generation.

This script shows how to use the event processing system to generate
spatio-temporal event tensors for Hong Kong data.
"""

import os
import sys
import numpy as np
import logging

# Add parent directory to path to import events module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events.tensor_generator import create_event_tensors

def main():
    """Main function to generate event tensors."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Configuration - use absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'data_HK')
    output_path = os.path.join(os.path.dirname(__file__), 'hk_events_2019.npz')
    year = 2019
    
    logger.info("Starting event tensor generation example...")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Year: {year}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
        return
    
    # Check required data files
    required_files = [
        'HK_regions/mtr_unique_stations.csv',
        'HK_regions/mtr_voronoi_final.geojson',
        'Events/mtr_events.txt',
        'Events/Hong_Kong_Concerts_2019_Final_processed.csv',
        'Events/weather_warnings_2019_processed.csv'
    ]
    
    for req_file in required_files:
        file_path = os.path.join(data_dir, req_file)
        if not os.path.exists(file_path):
            logger.error(f"Required file does not exist: {file_path}")
            return
        else:
            logger.info(f"Found required file: {req_file}")
    
    try:
        # Generate event tensors
        tensors = create_event_tensors(data_dir, output_path, year=year)
        
        if 'event_tensor' in tensors:
            event_tensor = tensors['event_tensor']
            metadata = tensors['metadata']
            statistics = tensors['statistics']
            
            # Print tensor information
            logger.info(f"Generated event tensor with shape: {event_tensor.shape}")
            logger.info(f"Tensor dimensions: {metadata['dimensions']}")
            logger.info(f"Channel descriptions:")
            for ch, desc in metadata['channel_descriptions'].items():
                logger.info(f"  Channel {ch}: {desc}")
            
            # Print statistics
            logger.info("\nTensor Statistics:")
            tensor_stats = statistics['tensor_statistics']
            logger.info(f"  Sparsity: {tensor_stats['sparsity']:.4f}")
            logger.info(f"  Active regions: {tensor_stats['active_regions']}")
            logger.info(f"  Active timesteps: {tensor_stats['active_timesteps']}")
            logger.info(f"  Max value: {tensor_stats['max_value']:.4f}")
            logger.info(f"  Mean non-zero: {tensor_stats['mean_nonzero']:.4f}")
            
            # Channel-wise statistics
            logger.info("\nChannel-wise Statistics:")
            for ch, stats in statistics['channel_statistics'].items():
                logger.info(f"  Channel {ch}: {stats['nonzero_count']} non-zero entries, "
                          f"max={stats['max_value']:.4f}, sparsity={stats['sparsity']:.4f}")
            
            logger.info(f"\nTensors saved to: {output_path}")
            
        else:
            logger.error("Failed to generate event tensors")
    
    except Exception as e:
        logger.error(f"Error generating event tensors: {e}")
        raise

if __name__ == "__main__":
    main()