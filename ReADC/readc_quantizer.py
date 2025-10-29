"""
ReADC: Reconfigurable Analog-to-Digital Converter with Adaptive Quantization

This module implements the core quantization functions for the ReADC system,
including both uniform and adaptive non-uniform quantization methods.

Author: Haiqiao Hong
Paper: "Memristor-based adaptive analog-to-digital conversion for efficient and accurate compute-in-memory"
"""

import torch
import numpy as np


def linear_quantize(input_tensor, bit_precision):
    """
    Uniform linear quantization function.
    
    Args:
        input_tensor (torch.Tensor): Input tensor to be quantized
        bit_precision (int): Number of bits for quantization
        
    Returns:
        torch.Tensor: Quantized tensor
    """
    min_val = torch.min(input_tensor)
    max_val = torch.max(input_tensor)
    delta = max_val - min_val
    
    quantized = input_tensor.clone()
    
    if delta > 0:
        step_size_ratio = 2.0 ** (-bit_precision)
        step_size = step_size_ratio * delta.item()
        
        # Quantize to discrete levels
        indices = torch.clamp(
            torch.floor((input_tensor - min_val.item()) / step_size), 
            0, 
            (2.0 ** bit_precision) - 1
        )
        quantized = indices * step_size + min_val.item()
    
    return quantized


def adaptive_quantize(input_tensor, bit_precision, boundaries=None, output_levels=None):
    """
    Adaptive non-uniform quantization with programmable boundaries.
    
    This is the core ReADC quantization function that enables adaptive quantization
    by using programmable memristor-based boundaries.
    
    Args:
        input_tensor (torch.Tensor): Input tensor to be quantized
        bit_precision (int): Number of bits for quantization (used for uniform fallback)
        boundaries (list, optional): Quantization boundaries. If None, falls back to uniform quantization
        output_levels (list, optional): Output quantization levels. If None, falls back to uniform quantization
        
    Returns:
        torch.Tensor: Quantized tensor
    """
    # Fallback to uniform quantization if boundaries are not provided
    if boundaries is None or output_levels is None:
        return linear_quantize(input_tensor, bit_precision)
    
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    # Convert boundaries and output levels to tensors
    # Add the last boundary value as the final output level
    output_tensor = torch.tensor(
        output_levels + [boundaries[-1]], 
        device=device, 
        dtype=torch.float64
    )
    
    boundary_tensor = torch.tensor(boundaries, device=device, dtype=dtype)
    
    # Add infinite bounds at the edges for proper binning
    boundary_tensor = torch.cat([
        torch.tensor([float('-inf')], device=device, dtype=dtype),
        boundary_tensor,
        torch.tensor([float('inf')], device=device, dtype=dtype)
    ])
    
    # Find the appropriate quantization bin for each input value
    indices = torch.searchsorted(boundary_tensor, input_tensor, right=True) - 1
    
    # Map indices to output levels
    quantized_output = output_tensor[indices]
    
    return quantized_output.to(dtype)


def multi_adc_quantize(input_tensor, bit_precision, boundary_sets=None, output_levels=None):
    """
    Multi-ADC quantization with super-resolution capability.
    
    This function implements the super-resolution strategy mentioned in the paper,
    where multiple ADCs with slight variations are used to improve quantization accuracy.
    
    Args:
        input_tensor (torch.Tensor): Input tensor to be quantized
        bit_precision (int): Number of bits for quantization
        boundary_sets (list of lists, optional): Multiple sets of boundaries for different ADCs
        output_levels (list, optional): Output quantization levels
        
    Returns:
        torch.Tensor: Quantized tensor using multi-ADC approach
    """
    if boundary_sets is None or output_levels is None:
        return linear_quantize(input_tensor, bit_precision)
    
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    # Prepare output levels tensor
    output_tensor = torch.tensor(
        output_levels + [boundary_sets[0][-1]], 
        device=device, 
        dtype=torch.float64
    )
    
    # Number of ADC sets (excluding the first reference set)
    num_adc_sets = len(boundary_sets) - 1
    
    # Store results from all ADC sets
    all_results = torch.zeros(num_adc_sets, *input_tensor.shape, dtype=torch.long, device=device)
    
    # Process each ADC set (starting from index 1)
    for i in range(1, len(boundary_sets)):
        boundary_tensor = torch.tensor(
            [float('-inf')] + boundary_sets[i] + [float('inf')], 
            device=device, 
            dtype=dtype
        )
        all_results[i-1] = torch.searchsorted(boundary_tensor, input_tensor, right=True) - 1
    
    # Use mode (most frequent value) across different ADC results
    result_indices = torch.mode(all_results, dim=0).values
    
    # Map to output levels
    quantized_output = output_tensor[result_indices].to(dtype)
    
    return quantized_output


def super_resolution_quantize(input_tensor, bit_precision, boundary_sets=None, output_levels=None):
    """
    Super-resolution quantization strategy.
    
    When two ADCs with nominally identical boundaries produce different outputs due to
    device variations, this function identifies boundary-proximate states and assigns
    appropriate boundary values, utilizing device variations for super-resolution.
    
    Args:
        input_tensor (torch.Tensor): Input tensor to be quantized
        bit_precision (int): Number of bits for quantization
        boundary_sets (list of lists): Multiple sets of boundaries (should have exactly 2 sets)
        output_levels (list): Output quantization levels
        
    Returns:
        torch.Tensor: Super-resolution quantized tensor
    """
    if boundary_sets is None or output_levels is None or len(boundary_sets) < 2:
        return linear_quantize(input_tensor, bit_precision)
    
    device = input_tensor.device
    dtype = input_tensor.dtype
    
    # Prepare output levels tensor
    output_tensor = torch.tensor(
        output_levels + [boundary_sets[0][-1]], 
        device=device, 
        dtype=torch.float64
    )
    
    # Get results from both ADC sets
    num_adc_sets = len(boundary_sets) - 1
    all_results = torch.zeros(num_adc_sets, *input_tensor.shape, dtype=torch.long, device=device)
    
    for i in range(1, len(boundary_sets)):
        boundary_tensor = torch.tensor(
            [float('-inf')] + boundary_sets[i] + [float('inf')], 
            device=device, 
            dtype=dtype
        )
        all_results[i-1] = torch.searchsorted(boundary_tensor, input_tensor, right=True) - 1
    
    # Check where results agree or disagree
    same_results = torch.all(all_results[0] == all_results[1:], dim=0)
    
    # For agreeing results, use normal output levels
    output = output_tensor[all_results[0]].to(dtype)
    
    # For disagreeing results, use boundary values (super-resolution)
    different_mask = ~same_results
    if torch.any(different_mask):
        min_indices = torch.min(all_results[:, different_mask], dim=0).values
        boundary_values = torch.tensor(
            [boundary_sets[0][idx] for idx in min_indices], 
            device=device, 
            dtype=dtype
        )
        output[different_mask] = boundary_values
    
    return output


# Alias for backward compatibility and ease of use
readc_quantize = adaptive_quantize
