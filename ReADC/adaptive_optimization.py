"""
Adaptive Quantization Optimization for ReADC

This module implements the Lloyd-Max algorithm and related optimization techniques
for determining optimal quantization boundaries for adaptive ADC quantization.

Author: Haiqiao Hong
Paper: "Memristor-based adaptive analog-to-digital conversion for efficient and accurate compute-in-memory"
"""

import numpy as np
import time
import os
import pickle


def lloyd_max_algorithm(data, num_bits, max_iterations=500, convergence_threshold=1e-6, time_limit=60):
    """
    Lloyd-Max algorithm for optimal quantization boundary determination.
    
    This algorithm iteratively optimizes quantization boundaries and output levels
    to minimize mean squared error (MSE) for the given data distribution.
    
    Args:
        data (np.ndarray): Input data array for optimization
        num_bits (int): Number of quantization bits
        max_iterations (int): Maximum number of iterations
        convergence_threshold (float): Relative MSE improvement threshold for convergence
        time_limit (int): Time limit in seconds
        
    Returns:
        tuple: (boundaries, output_levels) - Optimized quantization parameters
    """
    start_time = time.time()
    num_levels = 2 ** int(num_bits)
    
    # Initialize boundaries and output levels
    min_data, max_data = np.min(data), np.max(data)
    boundaries = np.linspace(min_data, max_data, num_levels + 1)
    output_levels = (boundaries[:-1] + boundaries[1:]) / 2
    
    mse = np.inf
    iteration = 0
    min_iterations = 100
    
    while iteration < max_iterations:
        # Check time limit
        if time.time() - start_time > time_limit:
            print(f'Time limit exceeded. Iterations: {iteration}, MSE: {mse}')
            break
        
        new_output_levels = output_levels.copy()
        new_boundaries = boundaries.copy()
        mse_new = 0
        
        # Update output levels (centroids)
        for i in range(len(output_levels)):
            mask = (data >= boundaries[i]) & (data < boundaries[i + 1])
            if np.any(mask):
                new_output_levels[i] = np.mean(data[mask])
                mse_new += np.sum((data[mask] - new_output_levels[i]) ** 2)
            else:
                new_output_levels[i] = output_levels[i]
        
        # Update boundaries (decision boundaries)
        new_boundaries[0] = min_data
        new_boundaries[-1] = max_data
        for i in range(1, len(output_levels)):
            new_boundaries[i] = (new_output_levels[i - 1] + new_output_levels[i]) / 2
        
        # Calculate MSE
        mse_new /= len(data)
        
        # Check convergence
        if iteration >= min_iterations:
            relative_improvement = np.abs(mse - mse_new) / (mse + 1e-10)
            if relative_improvement < convergence_threshold:
                print(f'Converged: MSE {mse_new:.6f}, Iterations: {iteration}')
                break
        
        # Update for next iteration
        boundaries = new_boundaries
        output_levels = new_output_levels
        mse = mse_new
        iteration += 1
    
    return boundaries, output_levels


def optimize_quantization_boundaries(data_file, num_bits, percentile_range=(0.5, 99.5), max_samples=8000):
    """
    Optimize quantization boundaries for given data file.
    
    Args:
        data_file (str): Path to data file (.npy or .csv)
        num_bits (int): Number of quantization bits
        percentile_range (tuple): Percentile range for data filtering
        max_samples (int): Maximum number of samples to use for optimization
        
    Returns:
        tuple: (boundaries, output_levels) - Optimized quantization parameters
    """
    # Load data
    if data_file.endswith('.csv'):
        data = np.loadtxt(data_file, delimiter=',', dtype=float)
    elif data_file.endswith('.npy'):
        data = np.load(data_file).flatten()
    else:
        raise ValueError(f"Unsupported file format: {data_file}")
    
    # Store original range
    min_value = np.min(data)
    max_value = np.max(data)
    
    # Filter data by percentiles to remove outliers
    lower_bound = np.percentile(data, percentile_range[0])
    upper_bound = np.percentile(data, percentile_range[1])
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    # Sample data if too large
    if len(filtered_data) > max_samples:
        filtered_data = np.random.choice(filtered_data, max_samples, replace=False)
    
    # Add original min/max to ensure full range coverage
    filtered_data = np.append(filtered_data, [min_value, max_value])
    
    # Run Lloyd-Max optimization
    boundaries, output_levels = lloyd_max_algorithm(filtered_data, num_bits)
    
    # Remove first and last boundaries (they are min/max values)
    # Keep only the internal decision boundaries
    optimized_boundaries = boundaries[1:-1].tolist() + [max_value]
    optimized_output_levels = output_levels.tolist()
    
    return optimized_boundaries, optimized_output_levels


def add_device_variation(boundaries, output_levels, sigma):
    """
    Add device variation to quantization boundaries to simulate memristor variations.
    
    Args:
        boundaries (list): Original quantization boundaries
        output_levels (list): Original output levels
        sigma (float): Standard deviation of variation (relative to data range)
        
    Returns:
        list: Boundaries with added variation
    """
    if sigma == 0:
        return boundaries
    
    min_value = boundaries[0] if boundaries else 0
    max_value = boundaries[-1] if boundaries else 1
    variation_std = sigma * (max_value - min_value)
    
    varied_boundaries = [boundaries[0]]  # Keep first boundary unchanged
    
    # Add variation to internal boundaries
    for i, boundary in enumerate(boundaries[1:-1], 1):
        left_neighbor = boundaries[i - 1]
        right_neighbor = boundaries[i + 1]
        
        # Generate variation within reasonable bounds
        while True:
            variation = np.random.normal(0, variation_std)
            if abs(variation) <= 2 * variation_std:
                new_boundary = boundary + variation
                varied_boundaries.append(new_boundary)
                break
    
    varied_boundaries.append(boundaries[-1])  # Keep last boundary unchanged
    
    return varied_boundaries


def batch_optimize_quantization(data_directory, model_name, num_bits=5, sigma=0, num_adc_variants=1, use_cache=True):
    """
    Batch optimization of quantization boundaries for multiple layers.
    
    Args:
        data_directory (str): Directory containing layer output data files
        model_name (str): Model name for caching
        num_bits (int): Number of quantization bits
        sigma (float): Device variation parameter
        num_adc_variants (int): Number of ADC variants for super-resolution
        use_cache (bool): Whether to use cached results
        
    Returns:
        tuple: (boundaries_dict, output_levels_dict) - Dictionaries keyed by layer identifier
    """
    boundaries_dict = {}
    output_levels_dict = {}
    
    # Create cache directory
    cache_dir = f'./cache_{model_name}'
    if use_cache and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Process each data file
    for filename in os.listdir(data_directory):
        if not (filename.startswith(("output", "x")) and filename.endswith((".npy", ".csv"))):
            continue
        
        # Parse layer information from filename
        if "FC" in filename:
            layer_type = "L"  # Linear layer
            parts = filename.split("FC")[1].split("_")
            layer_num = parts[1] if len(parts) > 2 else parts[0]
        elif "Conv" in filename:
            layer_type = "C"  # Convolutional layer
            parts = filename.split("Conv")[1].split("_")
            layer_num = parts[1] if len(parts) > 2 else parts[0]
        else:
            continue
        
        layer_id = (layer_type, layer_num)
        print(f'Processing layer {layer_id}: {filename}')
        
        # Check cache
        cache_file = os.path.join(cache_dir, f"{layer_type}_{layer_num}_ADC{num_bits}.pkl")
        if use_cache and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                print(f'Loading from cache: {cache_file}')
                boundaries, output_levels = pickle.load(f)
        else:
            # Optimize boundaries
            data_file = os.path.join(data_directory, filename)
            boundaries, output_levels = optimize_quantization_boundaries(data_file, num_bits)
            
            # Save to cache
            if use_cache:
                with open(cache_file, 'wb') as f:
                    pickle.dump((boundaries, output_levels), f)
        
        # Generate variants with device variation
        boundary_variants = [boundaries]  # First variant is the ideal one
        
        if sigma > 0:
            for _ in range(num_adc_variants):
                varied_boundaries = add_device_variation(boundaries, output_levels, sigma)
                boundary_variants.append(varied_boundaries)
        else:
            # Add identical copy for consistency
            boundary_variants.append(boundaries)
        
        # Round for numerical stability
        boundary_variants = [[round(b, 10) for b in variant] for variant in boundary_variants]
        output_levels_rounded = [round(ol, 10) for ol in output_levels]
        
        boundaries_dict[layer_id] = boundary_variants
        output_levels_dict[layer_id] = output_levels_rounded
    
    return boundaries_dict, output_levels_dict


def linear_quantization_boundaries(data_directory, num_bits=5, sigma=0):
    """
    Generate linear (uniform) quantization boundaries for comparison.
    
    Args:
        data_directory (str): Directory containing layer output data files
        num_bits (int): Number of quantization bits
        sigma (float): Device variation parameter
        
    Returns:
        tuple: (boundaries_dict, output_levels_dict) - Linear quantization parameters
    """
    boundaries_dict = {}
    output_levels_dict = {}
    
    for filename in os.listdir(data_directory):
        if not (filename.startswith(("output", "x")) and filename.endswith(".npy")):
            continue
        
        # Parse layer information
        if "FC" in filename:
            layer_type = "L"
            parts = filename.split("FC")[1].split("_")
            layer_num = parts[1] if len(parts) > 2 else parts[0]
        elif "Conv" in filename:
            layer_type = "C"
            parts = filename.split("Conv")[1].split("_")
            layer_num = parts[1] if len(parts) > 2 else parts[0]
        else:
            continue
        
        layer_id = (layer_type, layer_num)
        
        # Load data to get range
        data = np.load(os.path.join(data_directory, filename)).flatten()
        min_value = np.min(data)
        max_value = np.max(data)
        
        # Generate uniform boundaries
        num_levels = 2 ** num_bits
        boundaries = np.linspace(min_value, max_value, num_levels)
        output_levels = (boundaries[:-1] + boundaries[1:]) / 2
        
        # Convert to the expected format
        boundaries_list = boundaries[1:].tolist()  # Remove first boundary
        output_levels_list = output_levels.tolist()
        
        # Add device variation if specified
        if sigma > 0:
            boundaries_list = add_device_variation(boundaries_list, output_levels_list, sigma)
        
        boundaries_dict[layer_id] = boundaries_list
        output_levels_dict[layer_id] = output_levels_list
    
    return boundaries_dict, output_levels_dict
