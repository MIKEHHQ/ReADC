"""
ReADC: Reconfigurable Analog-to-Digital Converter

A Python package for adaptive quantization in compute-in-memory systems.
"""

from .readc_quantizer import (
    linear_quantize,
    adaptive_quantize,
    multi_adc_quantize,
    super_resolution_quantize,
    readc_quantize
)

from .adaptive_optimization import (
    lloyd_max_algorithm,
    optimize_quantization_boundaries,
    add_device_variation,
    batch_optimize_quantization,
    linear_quantization_boundaries
)

__version__ = "1.0.0"
__author__ = "Haiqiao Hong"
__email__ = "haiqiao@connect.hku.hk"

__all__ = [
    # Core quantization functions
    'linear_quantize',
    'adaptive_quantize', 
    'multi_adc_quantize',
    'super_resolution_quantize',
    'readc_quantize',
    
    # Optimization functions
    'lloyd_max_algorithm',
    'optimize_quantization_boundaries',
    'add_device_variation',
    'batch_optimize_quantization',
    'linear_quantization_boundaries'
]
