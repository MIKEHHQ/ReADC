#!/usr/bin/env python3
"""
Simple test script for ReADC quantization functions.

This script verifies that the core functionality works correctly.
"""

import torch
import numpy as np
from readc_quantizer import linear_quantize, adaptive_quantize, multi_adc_quantize, super_resolution_quantize
from adaptive_optimization import lloyd_max_algorithm


def test_linear_quantization():
    """Test uniform linear quantization."""
    print("Testing linear quantization...")
    
    # Create test data
    data = torch.linspace(0, 1, 100)
    
    # Test different bit precisions
    for bits in [2, 3, 4, 5]:
        result = linear_quantize(data, bits)
        
        # Check that result is within expected range
        assert torch.min(result) >= torch.min(data)
        assert torch.max(result) <= torch.max(data)
        
        # Check that we have the expected number of unique values
        unique_values = torch.unique(result)
        expected_levels = 2 ** bits
        assert len(unique_values) <= expected_levels
        
        print(f"  {bits}-bit: {len(unique_values)} unique levels (max: {expected_levels})")
    
    print("✓ Linear quantization tests passed")


def test_adaptive_quantization():
    """Test adaptive quantization."""
    print("\nTesting adaptive quantization...")
    
    data = torch.linspace(0, 1, 100)
    
    # Test with custom boundaries
    boundaries = [0.25, 0.5, 0.75, 1.0]
    levels = [0.125, 0.375, 0.625, 0.875]
    
    result = adaptive_quantize(data, 4, boundaries, levels)
    
    # Check basic properties
    assert torch.min(result) >= 0
    assert torch.max(result) <= 1
    
    # Test fallback to linear quantization
    fallback_result = adaptive_quantize(data, 4, None, None)
    linear_result = linear_quantize(data, 4)
    
    # Should be identical when no boundaries provided
    assert torch.allclose(fallback_result, linear_result, atol=1e-6)
    
    print("✓ Adaptive quantization tests passed")


def test_lloyd_max_optimization():
    """Test Lloyd-Max algorithm."""
    print("\nTesting Lloyd-Max optimization...")
    
    # Create non-uniform data
    np.random.seed(42)
    data = np.random.beta(2, 5, 1000)
    
    # Test optimization
    boundaries, levels = lloyd_max_algorithm(data, num_bits=3, max_iterations=50)
    
    # Check that we get reasonable results
    assert len(boundaries) == 2**3 + 1  # num_levels + 1
    assert len(levels) == 2**3  # num_levels
    
    # Boundaries should be sorted
    assert np.all(np.diff(boundaries) > 0)
    
    # Boundaries should span the data range
    assert boundaries[0] <= np.min(data)
    assert boundaries[-1] >= np.max(data)
    
    print(f"  Optimized {len(levels)} levels with MSE optimization")
    print("✓ Lloyd-Max optimization tests passed")


def test_multi_adc_quantization():
    """Test multi-ADC quantization."""
    print("\nTesting multi-ADC quantization...")
    
    data = torch.linspace(0, 1, 100)
    
    # Create boundary sets with slight variations
    boundary_sets = [
        [0.25, 0.5, 0.75, 1.0],
        [0.24, 0.51, 0.76, 1.0],
        [0.26, 0.49, 0.74, 1.0]
    ]
    levels = [0.125, 0.375, 0.625, 0.875]
    
    result = multi_adc_quantize(data, 4, boundary_sets, levels)
    
    # Check basic properties
    assert torch.min(result) >= 0
    assert torch.max(result) <= 1
    
    # Test fallback behavior
    fallback_result = multi_adc_quantize(data, 4, None, None)
    linear_result = linear_quantize(data, 4)
    assert torch.allclose(fallback_result, linear_result, atol=1e-6)
    
    print("✓ Multi-ADC quantization tests passed")


def test_super_resolution_quantization():
    """Test super-resolution quantization with device variations."""
    print("\nTesting super-resolution quantization...")
    
    data = torch.linspace(0, 1, 100)
    
    # Create two boundary sets with slight variations (simulating device variations)
    boundary_sets = [
        [0.25, 0.5, 0.75, 1.0],           # Reference boundaries
        [0.24, 0.51, 0.76, 1.0]           # Variant with device variations
    ]
    levels = [0.125, 0.375, 0.625, 0.875]
    
    result = super_resolution_quantize(data, 4, boundary_sets, levels)
    
    # Check basic properties
    assert torch.min(result) >= 0
    assert torch.max(result) <= 1
    
    # Test that super-resolution gives different results than single ADC
    single_adc_result = adaptive_quantize(data, 4, boundary_sets[0], levels)
    
    # Should have some differences due to super-resolution effect
    differences = torch.sum(torch.abs(result - single_adc_result) > 1e-6).item()
    print(f"  Super-resolution differences: {differences} out of {len(data)} points")
    
    # Test fallback behavior
    fallback_result = super_resolution_quantize(data, 4, None, None)
    linear_result = linear_quantize(data, 4)
    assert torch.allclose(fallback_result, linear_result, atol=1e-6)
    
    # Test with insufficient boundary sets
    single_set_result = super_resolution_quantize(data, 4, [boundary_sets[0]], levels)
    assert torch.allclose(single_set_result, linear_result, atol=1e-6)
    
    print("✓ Super-resolution quantization tests passed")


def test_noise_robustness():
    """Test noise robustness of super-resolution quantization."""
    print("\nTesting noise robustness...")
    
    # Create clean test data
    np.random.seed(42)
    clean_data = torch.tensor(np.random.beta(2, 3, 1000), dtype=torch.float32)
    
    # Optimize boundaries for clean data
    boundaries, levels = lloyd_max_algorithm(clean_data.numpy(), 4, max_iterations=50)
    adaptive_boundaries = boundaries[1:-1].tolist() + [1.0]
    adaptive_levels = levels.tolist()
    
    # Create boundary sets with device variations
    boundary_sets = [
        adaptive_boundaries,
        [b + 0.005 for b in adaptive_boundaries[:-1]] + [1.0],  # Small variation
        [b - 0.005 for b in adaptive_boundaries[:-1]] + [1.0]   # Small variation
    ]
    
    # Test with different noise levels
    noise_levels = [0.01, 0.05]  # 1% and 5% noise
    
    for noise_level in noise_levels:
        print(f"  Testing with {noise_level*100}% noise...")
        
        # Add noise to data
        noise = torch.randn_like(clean_data) * noise_level * torch.std(clean_data)
        noisy_data = clean_data + noise
        
        # Compare different quantization methods
        uniform_result = linear_quantize(noisy_data, 4)
        adaptive_result = adaptive_quantize(noisy_data, 4, adaptive_boundaries, adaptive_levels)
        sr_result = super_resolution_quantize(noisy_data, 4, boundary_sets, adaptive_levels)
        
        # Calculate MSE relative to clean data
        uniform_mse = torch.mean((clean_data - uniform_result) ** 2).item()
        adaptive_mse = torch.mean((clean_data - adaptive_result) ** 2).item()
        sr_mse = torch.mean((clean_data - sr_result) ** 2).item()
        
        print(f"    Uniform MSE: {uniform_mse:.6f}")
        print(f"    Adaptive MSE: {adaptive_mse:.6f}")
        print(f"    Super-Resolution MSE: {sr_mse:.6f}")
        
        # Super-resolution should be robust to noise
        sr_improvement = ((adaptive_mse - sr_mse) / adaptive_mse * 100) if adaptive_mse > 0 else 0
        print(f"    SR improvement over adaptive: {sr_improvement:.2f}%")
        
        # Basic sanity checks
        assert sr_mse <= uniform_mse * 1.2  # SR should be better than or comparable to uniform
        assert not torch.isnan(sr_result).any()  # No NaN values
        
    print("✓ Noise robustness tests passed")


def test_quantization_accuracy():
    """Test quantization accuracy comparison."""
    print("\nTesting quantization accuracy...")
    
    # Create test data with known distribution
    np.random.seed(42)
    data = torch.tensor(np.random.beta(2, 3, 1000), dtype=torch.float32)
    
    # Compare uniform vs adaptive quantization
    uniform_result = linear_quantize(data, 4)
    uniform_mse = torch.mean((data - uniform_result) ** 2).item()
    
    # Optimize boundaries for this data
    boundaries, levels = lloyd_max_algorithm(data.numpy(), 4, max_iterations=100)
    adaptive_boundaries = boundaries[1:-1].tolist() + [1.0]
    adaptive_levels = levels.tolist()
    
    adaptive_result = adaptive_quantize(data, 4, adaptive_boundaries, adaptive_levels)
    adaptive_mse = torch.mean((data - adaptive_result) ** 2).item()
    
    print(f"  Uniform MSE: {uniform_mse:.6f}")
    print(f"  Adaptive MSE: {adaptive_mse:.6f}")
    print(f"  Improvement: {((uniform_mse - adaptive_mse) / uniform_mse * 100):.1f}%")
    
    # Adaptive should generally be better or equal
    assert adaptive_mse <= uniform_mse * 1.1  # Allow small tolerance
    
    print("✓ Quantization accuracy tests passed")


def main():
    """Run all tests."""
    print("ReADC Quantization Test Suite")
    print("=" * 40)
    
    try:
        test_linear_quantization()
        test_adaptive_quantization()
        test_lloyd_max_optimization()
        test_multi_adc_quantization()
        test_super_resolution_quantization()
        test_noise_robustness()
        test_quantization_accuracy()
        
        print("\n" + "=" * 40)
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
