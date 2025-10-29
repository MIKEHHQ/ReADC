# ReADC: Reconfigurable Analog-to-Digital Converter and High Performance ADC Simulator

This repository contains the implementation of ReADC (Reconfigurable Analog-to-Digital Converter) with adaptive quantization for efficient and accurate compute-in-memory systems, along with a high-performance parallel ADC simulator.

## Paper Reference

**"Memristor-based adaptive analog-to-digital conversion for efficient and accurate compute-in-memory"**
Authors: Haiqiao Hong, Zhiyuan Du, Mingrui Jiang, Ruibin Mao, Yuan Ren, Fuyi Li, Wei Mao, Muyuan Peng, Wei Zhang, Zhengwu Liu, Can Li, Ngai Wong
Published in Nature Communications
DOI: https://doi.org/10.1038/s41467-025-65233-w

## Overview

ReADC addresses the critical bottleneck of analog-to-digital conversion in compute-in-memory (CIM) systems by introducing:

1. **Adaptive Quantization**: Hardware-friendly adaptive quantization using programmable memristor boundaries
2. **Super-Resolution Strategy**: Utilizing device variations to improve quantization accuracy
3. **High-Performance Simulator**: Parallel ADC simulation framework that dramatically accelerates CIM system simulation
4. **Hardware Efficiency**: 15.1x improvement in energy efficiency and 12.9x reduction in area compared to state-of-the-art designs

## Key Innovations

### Hardware Design

- **Programmable Quantization Boundaries**: Memristor-based analog content-addressable memory (CAM) cells enable dynamic boundary reconfiguration
- **Lloyd-Max Optimization**: Optimal quantization boundary determination for minimizing mean squared error
- **Multi-ADC Super-Resolution**: Leveraging device variations for enhanced quantization precision

### High-Performance Simulator

Traditional compute-in-memory simulators suffer from significant performance bottlenecks when simulating ADC quantization, especially for adaptive quantization schemes. Our high-performance parallel ADC simulator addresses this critical limitation:

- **Parallel Processing**: Vectorized operations enable simultaneous quantization of multiple data points
- **Optimized Algorithms**: Efficient implementation of Lloyd-Max optimization with caching mechanisms
- **Scalable Architecture**: Support for batch processing of multiple layers and network architectures
- **Adaptive Quantization Acceleration**: Specialized optimizations for non-uniform quantization that traditionally slows down simulation

This simulator enables practical evaluation of large-scale neural networks with adaptive quantization, making it feasible to explore the full potential of ReADC in realistic CIM scenarios.

## Repository Structure

```
ReADC/
├── ReADC/                      # Core implementation
│   ├── readc_quantizer.py      # Core quantization functions
│   ├── adaptive_optimization.py # Lloyd-Max algorithm and optimization
│   ├── demo_quantization.ipynb # Usage demonstration and examples
│   ├── test_readc.py           # Comprehensive test suite
│   └── __init__.py             # Package initialization
├── Data/                       # Experimental data and results
├── Paper/                      # Paper-related materials
└── README.md                   # This file
```

## Installation

1. Clone this repository:

   ```bash

   ```

git clone https://github.com/MIKEHHQ/READC.git
   cd ReADC

```

2. Install dependencies:

```bash
pip install torch numpy matplotlib scipy jupyter
```

## Quick Start

### Basic Usage

```python
import torch
from ReADC.readc_quantizer import linear_quantize, adaptive_quantize

# Create sample data
data = torch.randn(100)

# Uniform quantization
uniform_result = linear_quantize(data, bit_precision=4)

# Adaptive quantization with custom boundaries
boundaries = [0.25, 0.5, 0.75, 1.0]
levels = [0.125, 0.375, 0.625, 0.875]
adaptive_result = adaptive_quantize(data, 4, boundaries, levels)
```

### Optimizing Quantization Boundaries

```python
import numpy as np
from ReADC.adaptive_optimization import lloyd_max_algorithm

# Generate sample data with non-uniform distribution
data = np.random.beta(2, 5, 1000)

# Optimize boundaries using Lloyd-Max algorithm
boundaries, levels = lloyd_max_algorithm(data, num_bits=4)
```

### Super-Resolution Quantization

```python
from ReADC.readc_quantizer import super_resolution_quantize

# Define multiple boundary sets (simulating device variations)
boundary_sets = [
    [0.25, 0.5, 0.75, 1.0],           # Reference boundaries
    [0.24, 0.51, 0.76, 1.0],          # Variant 1 with device variations
    [0.26, 0.49, 0.74, 1.0]           # Variant 2 with device variations
]
levels = [0.125, 0.375, 0.625, 0.875]

# Apply super-resolution quantization
result = super_resolution_quantize(data, 4, boundary_sets, levels)
```

## Demo and Examples

Run the Jupyter notebook `ReADC/demo_quantization.ipynb` for comprehensive examples including:

- Comparison between uniform and adaptive quantization
- Lloyd-Max algorithm demonstration
- Super-resolution quantization examples
- Performance analysis across different bit precisions

## API Reference

### Core Functions

#### `linear_quantize(input_tensor, bit_precision)`

Performs uniform linear quantization with high-performance parallel processing.

#### `adaptive_quantize(input_tensor, bit_precision, boundaries=None, output_levels=None)`

Performs adaptive non-uniform quantization with programmable boundaries. Falls back to uniform quantization when boundaries are not provided.

#### `super_resolution_quantize(input_tensor, bit_precision, boundary_sets=None, output_levels=None)`

Super-resolution quantization strategy that utilizes device variations to improve quantization accuracy.

#### `lloyd_max_algorithm(data, num_bits, max_iterations=500, convergence_threshold=1e-6, time_limit=60)`

Lloyd-Max algorithm for optimal quantization boundary determination with performance optimizations.

## Performance Results

Based on our experiments:

- **VGG8 on CIFAR-10**: 89.55% accuracy at 5-bit adaptive quantization
- **ResNet18 on ImageNet**: 65.50% accuracy at 6-bit with super-resolution
- **Hardware Efficiency**: 15.1x energy improvement, 12.9x area reduction
- **System Integration**: Up to 57.2% energy overhead reduction in CIM systems
- **Simulation Speedup**: Orders of magnitude faster ADC simulation compared to traditional approaches

## Integration with NeuroSim

ReADC can be integrated with NeuroSim-based compute-in-memory simulators to provide high-performance adaptive quantization. Below is an example of how to integrate ReADC quantization functions into existing CIM simulation frameworks.

### Integration Example

The main integration point is in the forward pass of NeuroSim's quantized layers. Here's how to replace the original ADC quantization calls:

```python
# In NeuroSim's QLinear forward method (around line 583 in quantization_cpu_np_infer.py)
from ReADC.readc_quantizer import adaptive_quantize, super_resolution_quantize

class QLinear(nn.Linear):
    def forward(self, input):
        # ... existing NeuroSim code for weight processing ...
      
        if self.inference == 3:  # Uniform Quantization mode
            output = F.linear(input, weight, self.bias)
            with torch.no_grad():
                # Replace this line:
                # output = wage_quantizer.ReNonLinearQuantizeOut(output, self.ADCprecision, self.bound, self.out)
              
                # With ReADC function:
                output = adaptive_quantize(output, self.ADCprecision, self.bound, self.out)
              
        elif self.inference == 4:  # Adaptive Quantization mode  
            output = F.linear(input, weight, self.bias)
            with torch.no_grad():
                # Replace this line:
                # output = wage_quantizer.MultiReADC(output, self.ADCprecision, self.bound, self.out)
              
                # With ReADC super-resolution function:
                output = super_resolution_quantize(output, self.ADCprecision, self.bound, self.out)
      
        # ... rest of existing NeuroSim code ...
        return output

# Similar replacement applies to QConv2d layers
```

### Boundary Optimization Setup

```python
from ReADC.adaptive_optimization import batch_optimize_quantization

# Optimize quantization boundaries using collected activation data
boundaries_dict, output_levels_dict = batch_optimize_quantization(
    data_directory='./activation_VGG8/',  # Directory with collected .npy files
    model_name='VGG8',
    num_bits=5,
    sigma=0.01  # Device variation parameter
)

# Set optimized boundaries to NeuroSim layers
for name, module in model.named_modules():
    if isinstance(module, (QConv2d, QLinear)):
        layer_id = parse_layer_id(name)  # Extract layer identifier
        if layer_id in boundaries_dict:
            module.bound = boundaries_dict[layer_id][0]  # Boundaries
            module.out = output_levels_dict[layer_id]    # Output levels
```

### Key Integration Points

1. **Replace Traditional ADC Quantization**: Substitute uniform quantization calls with ReADC adaptive quantization functions
2. **Boundary Optimization**: Use `batch_optimize_quantization()` to determine optimal boundaries for each layer
3. **Device Variation Modeling**: Incorporate memristor variations through the `sigma` parameter
4. **Super-Resolution Enhancement**: Leverage multiple boundary sets for improved accuracy under device variations

### Performance Benefits in CIM Simulation

- **Simulation Speedup**: 10-100x faster than traditional iterative quantization methods
- **Memory Efficiency**: Vectorized operations reduce memory overhead
- **Accuracy Improvement**: Up to 37% MSE reduction with noise robustness
- **Scalability**: Efficient batch processing for large neural networks

### Copyright and Availability

Due to copyright considerations, we do not provide the complete modified NeuroSim codebase. The integration examples above demonstrate the key concepts for incorporating ReADC into existing CIM simulators. For specific integration questions or access to modified NeuroSim components, please contact the authors via email.

## Testing

Run the comprehensive test suite:

```bash
cd ReADC
python test_readc.py
```

The test suite includes:

- Basic quantization functionality tests
- Lloyd-Max optimization validation
- Super-resolution quantization with noise robustness testing
- Performance benchmarking across different configurations

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{hong2025memristor,
  title={Memristor-based adaptive analog-to-digital conversion for efficient and accurate compute-in-memory},
  author={Hong, Haiqiao and Du, Zhiyuan and Jiang, Mingrui and Mao, Ruibin and Ren, Yuan and Li, Fuyi and Mao, Wei and Peng, Muyuan and Zhang, Wei and Liu, Zhengwu and Li, Can and Wong, Ngai},
  journal={Nature Communications},
  year={2025},
  doi={10.1038/s41467-025-65233-w}
}
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. See the paper for full license details.

## Contact

For questions or collaborations, please contact:

- Haiqiao Hong: haiqiao@connect.hku.hk
- Zhengwu Liu: zwliu@eee.hku.hk
- Can Li: canl@hku.hk
- Ngai Wong: nwong@eee.hku.hk

## Acknowledgments

This work was supported by the Theme-based Research Scheme (TRS) project T45-701/22-R, the National Natural Science Foundation of China, Croucher Foundation, and the General Research Fund (GRF) projects of the Research Grants Council (RGC), Hong Kong SAR.

We would like to express our gratitude to Professor Shanshi Huang for the valuable email communications during the early stages of this project.
