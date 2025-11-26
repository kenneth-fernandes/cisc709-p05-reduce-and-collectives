# P05: Reduction & Collectives

**Course:** CISC-709 - Contemporary Computing Systems Programming (CCSP)  
**Assignment:** P05 - REDUCE & Collectives  
**Topic:** Associativity, Tree vs. Warp Reductions, Shuffles/Atomics; NCCL  
**Author:** Kenneth Peter Fernandes  
**Date:** November 26, 2025  

---

## Overview

This project implements and analyzes parallel reduction operations across multiple levels of the GPU computing stack:

1. **Custom CUDA Reduction Kernel** - Low-level warp shuffle and shared memory reduction
2. **Multi-GPU Distributed Reduction** - NCCL-based collective operations with PyTorch DDP
3. **Floating-Point Associativity Analysis** - Numerical accuracy investigation across reduction methods

The project demonstrates high-performance GPU programming techniques, distributed computing with NCCL, and the fundamental challenges of floating-point arithmetic in parallel systems.

---

## Key Results

### Task 1: Custom CUDA Reduction
- ✅ **Peak Throughput**: 252.71 GB/s (79% of T4 theoretical peak)
- ✅ **Speedup over Thrust**: 1.16×–2.57× across problem sizes
- ✅ **All Correctness Tests Passed**: 6/6 validation tests

### Task 2: Multi-GPU Reduction
- ✅ **Platform**: 2× NVIDIA Tesla T4 GPUs with NCCL backend
- ✅ **Correctness**: Bit-exact match with single-GPU reference
- ✅ **Scaling**: Demonstrated communication-computation tradeoff

### Task 3: Associativity Analysis
- ✅ **Error Range**: 3.24e-05 to 9.07e+13 depending on data distribution
- ✅ **Methods Tested**: 6 reduction algorithms (sequential, reverse, tree, sorted, NumPy, GPU)
- ✅ **Insights**: Quantified impact of reduction order on numerical accuracy

---

## Project Structure

```
cisc709-p05-reduce-and-collectives/
│
├── README.md                       # This file
│
├── cuda_code/                      # CUDA kernel implementations
│   ├── reduce_kernel.cu           # Custom reduction kernel (warp shuffles + shared memory)
│   └── thrust_reduce.cu           # Thrust baseline wrapper
│
├── notebooks/                      # Jupyter notebooks (main deliverables)
│   ├── task_1.ipynb               # Custom CUDA Reduction (17 cells)
│   ├── task_2.ipynb               # Multi-GPU Reduction with DDP (23 cells)
│   └── task_3.ipynb               # Floating-Point Associativity (19 cells)
│
└── reports/                        # Analysis and documentation
    └── reduce_analysis.md         # Comprehensive technical report (23KB)
```

---

## Prerequisites

### Hardware Requirements
- **Minimum**: 1× NVIDIA GPU with compute capability ≥ 7.0 (Tesla T4, RTX 2060+)
- **Recommended for Task 2**: 2× NVIDIA GPUs for multi-GPU experiments
- **Memory**: ≥ 8GB GPU memory per device

### Software Requirements

**CUDA Toolkit:**
```bash
CUDA >= 11.0
nvcc compiler
```

**Python Environment:**
```bash
python >= 3.8
numpy
torch >= 1.10 (with CUDA support)
matplotlib
numba
```

**Installation:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib numba
```

**For Multi-GPU (Task 2):**
```bash
# NCCL is included with PyTorch
# Verify multi-GPU setup:
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

---

## Running the Experiments

### Task 1: Custom CUDA Reduction

**Objective**: Implement and benchmark custom CUDA reduction kernel

**Steps:**

1. **Open Notebook**:
   ```bash
   jupyter notebook notebooks/task_1.ipynb
   ```

2. **Run All Cells** (in order):
   - Cell 0: Header and overview
   - Cell 1-2: Environment setup and GPU verification
   - Cell 3-4: CUDA kernel implementation (writes `reduce_kernel.cu`)
   - Cell 5-6: Thrust baseline (writes `thrust_reduce.cu`)
   - Cell 7-8: Compile kernels with `nvcc`
   - Cell 9-10: Python wrapper class
   - Cell 11-12: Correctness validation (6 tests)
   - Cell 13-14: Performance benchmarking
   - Cell 15-16: Visualization

3. **Expected Outputs**:
   - ✓ All 6 correctness tests pass
   - Performance table showing Custom vs Thrust vs NumPy
   - Plots: Execution time, throughput, speedup, efficiency

**Key Metrics:**
- Peak throughput: ~252 GB/s
- Memory efficiency: ~79%
- Speedup over Thrust: 1.5×–2.5×

---

### Task 2: Multi-GPU Reduction

**Objective**: Distributed reduction using PyTorch DDP and NCCL

**Requirements**: 2 GPUs (modify code for single-GPU testing if needed)

**Steps:**

1. **Verify Multi-GPU Setup**:
   ```bash
   nvidia-smi  # Should show 2 GPUs
   ```

2. **Open Notebook**:
   ```bash
   jupyter notebook notebooks/task_2.ipynb
   ```

3. **Run All Cells** (in order):
   - Cell 0: Header and overview
   - Cell 1-2: GPU memory cleanup
   - Cell 3-4: Environment verification
   - Cell 5-6: Multi-GPU availability check
   - Cell 7-8: Single-GPU baseline
   - Cell 9-10: Multi-GPU script (writes `multi_gpu_reduce.py`)
   - Cell 11-12: Execute multi-GPU with `torchrun`
   - Cell 13-14: Performance comparison
   - Cell 15-16: Correctness validation
   - Cell 17-18: Visualization
   - Cell 19-20: Communication analysis
   - Cell 21-22: Scaling analysis

4. **Expected Outputs**:
   - ✓ Correctness validation passes (0.00e+00 error)
   - Comparison table: Single vs Multi-GPU
   - Plots: Time, speedup, communication breakdown

**Note**: If only 1 GPU available, the notebook will show a warning. You can still run it, but multi-GPU cells will fail gracefully.

---

### Task 3: Associativity Experiments

**Objective**: Demonstrate non-associativity of floating-point arithmetic

**Steps:**

1. **Open Notebook**:
   ```bash
   jupyter notebook notebooks/task_3.ipynb
   ```

2. **Run All Cells** (in order):
   - Cell 0: Header and overview
   - Cell 1-2: Environment setup
   - Cell 3-4: Imports and configuration
   - Cell 5-6: Reduction method implementations
   - Cell 7-8: Data distribution generators
   - Cell 9-10: Run associativity experiments (may take 1-2 minutes)
   - Cell 11-12: GPU vs CPU comparison
   - Cell 13-14: Detailed results table
   - Cell 15-16: Error scaling visualization
   - Cell 17-18: Simple 3-number demonstration

3. **Expected Outputs**:
   - Error analysis for 3 data types (mixed, pathological, normal)
   - GPU vs CPU comparison
   - Error scaling plots
   - Demonstration: `(a+b)+c ≠ a+(b+c)` with 3 numbers

**Key Findings:**
- Error ranges from 3e-05 (normal data) to 9e+13 (mixed data)
- Sorted summation reduces error for pathological cases
- Tree reduction (GPU) has different error than sequential (CPU)

---

## Implementation Details

### Task 1: CUDA Kernel Architecture

**Three-Level Reduction Hierarchy:**

1. **Thread Level**: Grid-stride loop
   ```cuda
   for (int i = idx; i < n; i += gridSize) {
       sum += input[i];
   }
   ```

2. **Warp Level**: Shuffle-based reduction (no shared memory)
   ```cuda
   for (int offset = 16; offset > 0; offset >>= 1) {
       val += __shfl_down_sync(0xffffffff, val, offset);
   }
   ```

3. **Block Level**: Shared memory (32 floats) + final warp reduction
   ```cuda
   __shared__ float shared[32];  // One per warp
   if (laneId == 0) shared[warpId] = sum;
   __syncthreads();
   ```

4. **Grid Level**: Two-phase reduction (no atomics)
   - Phase 1: Each block reduces to one value
   - Phase 2: Single block reduces all block results

**Benefits:**
- Minimal shared memory usage (128 bytes)
- No bank conflicts
- Handles arbitrary input sizes
- Near-optimal memory bandwidth utilization

---

### Task 2: NCCL Configuration

**Distributed Setup:**

```python
import torch.distributed as dist

dist.init_process_group(backend='nccl')
rank = dist.get_rank()        # 0 or 1 for 2 GPUs
world_size = dist.get_world_size()  # 2

# Each GPU processes N/2 elements
local_data = torch.randn(size // world_size, device=f'cuda:{rank}')
local_sum = torch.sum(local_data)

# All-reduce across GPUs
dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
```

**Execution:**
```bash
torchrun --nproc_per_node=2 multi_gpu_reduce.py <array_size>
```

**NCCL automatically selects**:
- Ring or Tree algorithm based on message size
- Optimal GPU-to-GPU communication path
- PCIe or NVLink (if available)

---

### Task 3: Reduction Methods Tested

1. **Sequential**: `((a₁ + a₂) + a₃) + ... + aₙ`
2. **Reverse**: `a₁ + (a₂ + (a₃ + ... (aₙ)))`
3. **Tree**: Binary tree (GPU-style)
4. **Sorted**: Ascending magnitude order
5. **NumPy**: Optimized SIMD (CPU)
6. **PyTorch**: CUDA kernel (GPU)

**Data Distributions:**
- **Mixed**: Large (1e6–1e8) + Small (1e-3–1e-1) values
- **Pathological**: [1e8, 1.0, -1e8, ...] pattern
- **Normal**: N(0,1) distribution

---

## Performance Summary

### Task 1: Custom CUDA Reduction

| Metric | Value |
|--------|-------|
| Peak Throughput | 252.71 GB/s |
| Memory Efficiency | 79.0% of theoretical peak |
| Best Speedup vs Thrust | 2.57× |
| Best Speedup vs NumPy | 21.92× |
| Array Size for Peak | 100M elements |

**Conclusion**: Memory-bound kernel approaching hardware limits.

---

### Task 2: Multi-GPU Scaling

| Array Size | Single GPU (ms) | Multi-GPU (ms) | Speedup |
|------------|-----------------|----------------|---------|
| 1M | 0.0885 | 2.3022 | 0.04× (overhead) |
| 10M | 0.1833 | 0.3002 | 0.61× |
| 100M | 1.5028 | 0.3178 | 4.73× |
| 1B | 14.6031 | 0.3073 | 47.52× |

**Note**: Super-linear speedup at large sizes suggests measurement artifacts. See report for detailed analysis.

**Key Insight**: Communication overhead dominates for small arrays; large arrays amortize cost.

---

### Task 3: Floating-Point Error Summary

**Normal Data (typical case):**
- Sequential: 3.49e-01 max error
- Tree: 1.10e-03 max error
- NumPy: 2.01e-03 max error

**Pathological Data (worst case):**
- All methods: 1.67e+07 max error (except sorted: 0.00)
- **Sorted sum**: Perfect accuracy by avoiding cancellation

**Mixed Data:**
- Sequential: 7.61e+13 max error
- Tree: Varies (used as reference)
- Sorted: 9.07e+13 max error (counterintuitive!)

**Conclusion**: Reduction order significantly impacts accuracy; sorted helps pathological cases but not always.

---

## Validation and Testing

### Correctness Tests Implemented

**Task 1:**
- [x] Small known sum (100 ones)
- [x] Non-power-of-two size (1337 elements)
- [x] All zeros
- [x] Negative numbers
- [x] Large random array vs NumPy
- [x] Custom vs Thrust

**Task 2:**
- [x] Bit-exact validation vs single-GPU
- [x] Relative error < 1e-6

**Task 3:**
- [x] Multiple reduction methods
- [x] Three data distributions
- [x] Double-precision reference
- [x] GPU vs CPU comparison

---

## Known Issues and Limitations

### Task 1
- **Timing method**: Uses `time.time()` instead of CUDA events (less precise)
- **No Nsight profiling**: Missing occupancy and warp efficiency metrics
- **Single precision only**: Could extend to double precision

### Task 2
- **Anomalous speedup**: Super-linear speedup (>2×) indicates measurement issues
- **Timing methodology**: Should use CUDA events for GPU-accurate timing
- **Communication analysis**: Negative overhead suggests baseline mismatch
- **Recommendations**: See report section 2.6 for detailed improvements

### Task 3
- **Float32 only**: Could extend to float16, float64
- **No Kahan summation**: Missing compensated summation comparison
- **Limited array sizes**: Only up to 50M elements

---

## Report

A comprehensive 23KB technical report is available in `reports/reduce_analysis.md` covering:

- Implementation details for all three tasks
- Complete performance analysis with tables and insights
- Error analysis and numerical accuracy discussion
- Critical evaluation of results (including Task 2 measurement issues)
- Future work recommendations
- References and hardware specifications

**Read the report for**:
- Design decisions and rationale
- Detailed performance breakdowns
- Numerical accuracy analysis
- Lessons learned and best practices

---

## Key Takeaways

### Technical Achievements

1. **High-Performance Kernel**: Custom reduction achieves 79% memory bandwidth efficiency
2. **Distributed Computing**: Successfully implemented NCCL-based multi-GPU reduction
3. **Numerical Analysis**: Quantified floating-point error across multiple scenarios

### Performance Insights

- **Memory bandwidth is the limit**: At large sizes, reduction is memory-bound
- **Communication overhead matters**: Multi-GPU only beneficial for large problems
- **Reduction order affects accuracy**: Tree (GPU) ≠ Sequential (CPU) numerically

### Best Practices Learned

1. Use warp shuffles to minimize shared memory
2. Handle arbitrary sizes with grid-stride loops
3. Validate with high-precision reference (float64)
4. Profile with GPU-aware timing (CUDA events)
5. Test with pathological data to expose worst-case behavior

---

## Running in Different Environments

### Google Colab

```python
# Check GPU availability
!nvidia-smi

# Install required packages (if needed)
!pip install torch torchvision torchaudio

# Upload notebooks and run
```

### Kaggle

1. Enable GPU: Settings → Accelerator → GPU T4 x2 (for Task 2)
2. Upload notebooks to Kaggle kernel
3. Run cells in order

### Local Machine

```bash
# Clone or download project
cd cisc709-p05-reduce-and-collectives

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy matplotlib numba jupyter

# Launch Jupyter
jupyter notebook
```

---

## Troubleshooting

### "CUDA not available"
**Solution**: Install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### "nvcc: command not found"
**Solution**: Install CUDA Toolkit or ensure it's in PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

### "Only 1 GPU detected" (Task 2)
**Solution**:
- Kaggle: Enable "GPU T4 x2" in accelerator settings
- Colab: Free tier only has 1 GPU; upgrade to Colab Pro
- Local: Check `nvidia-smi` output

### Kernel compilation errors
**Solution**: Ensure compute capability matches:
```bash
nvcc -arch=sm_75  # For Tesla T4 (compute 7.5)
nvcc -arch=sm_80  # For A100 (compute 8.0)
```

---

## License

This project is submitted as coursework for CISC-709 (Fall 2025) at Hampton University.

---
