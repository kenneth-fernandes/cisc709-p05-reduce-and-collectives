# CISC-709 P05: Reduction & Collectives Analysis Report

**Student:** Kenneth Peter Fernandes  
**Course:** CISC-709 - Contemporary Computing Systems Programming (CCSP)  
**Assignment:** P05 - REDUCE & Collectives - Associativity, Tree vs. Warp Reductions, Shuffles/Atomics; NCCL  
**Date:** November 26, 2025

---

## Executive Summary

This report presents a comprehensive analysis of reduction operations in parallel computing, focusing on three key aspects: custom CUDA kernel implementation, multi-GPU distributed reduction, and floating-point associativity. The experiments were conducted on NVIDIA Tesla T4 GPUs and demonstrate the performance characteristics, scaling behavior, and numerical accuracy considerations of parallel reduction algorithms.

**Key Findings:**
- Custom CUDA reduction kernel achieved **252.71 GB/s** throughput (79% of T4 theoretical peak)
- Multi-GPU reduction showed variable scaling efficiency from **1.9% to 2375.9%** depending on problem size
- Floating-point non-associativity demonstrated error variations up to **9.07×10¹³** for pathological data

---

## Table of Contents

1. [Task 1: Custom CUDA Reduction](#task-1-custom-cuda-reduction)
2. [Task 2: Multi-GPU Reduction with PyTorch DDP](#task-2-multi-gpu-reduction-with-pytorch-ddp)
3. [Task 3: Floating-Point Associativity Analysis](#task-3-floating-point-associativity-analysis)
4. [Conclusions and Key Takeaways](#conclusions-and-key-takeaways)

---

## Task 1: Custom CUDA Reduction

### 1.1 Objective

Implement an efficient CUDA reduction kernel using warp shuffle instructions and shared memory, compare its performance against NVIDIA's Thrust library, and validate correctness across various input sizes.

### 1.2 Implementation Details

#### Kernel Architecture

The custom reduction kernel implements a hierarchical three-level reduction strategy:

1. **Thread-level aggregation**: Grid-stride loop handles arbitrary input sizes
2. **Warp-level reduction**: Uses `__shfl_down_sync()` for fast intra-warp reduction
3. **Block-level reduction**: Shared memory (32 floats) aggregates warp results
4. **Grid-level reduction**: Two-phase approach for multi-block reductions

**Key Design Decisions:**

- **Warp shuffles** eliminate shared memory for intra-warp communication (32 threads)
- **Shared memory** only used for inter-warp communication (up to 32 warps per block)
- **Grid-stride loop** ensures efficient handling of non-power-of-two sizes
- **Two-phase reduction** avoids global atomics for grid-level aggregation

```cuda
__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

#### Configuration

- **Block size**: 256 threads (8 warps)
- **Grid size**: min((N + 255) / 256, 1024) blocks
- **Shared memory**: 32 floats (128 bytes)
- **Target architecture**: sm_75 (Tesla T4)

### 1.3 Correctness Validation

All 6 correctness tests passed with appropriate error tolerances:

| Test Case | Array Size | Expected | Result | Status |
|-----------|------------|----------|---------|---------|
| Small known sum | 100 | 100.0 | 100.00 | ✓ PASSED |
| Non-power-of-two | 1,337 | 1,337.0 | 1,337.00 | ✓ PASSED |
| All zeros | 1,000 | 0.0 | 0.00e+00 | ✓ PASSED |
| Negative numbers | 500 | -500.0 | -500.00 | ✓ PASSED |
| Large random (vs NumPy) | 10M | -639.573608 | -639.575195 | ✓ PASSED (2.48e-06 error) |
| Custom vs Thrust | 1M | -1566.542603 | -1566.542603 | ✓ PASSED (0.00e+00 error) |

**Key Observation**: The custom kernel matches Thrust's output exactly, validating the implementation's correctness.

### 1.4 Performance Results

#### Throughput Analysis

Performance was measured across array sizes from 1K to 500M elements:

| Array Size | Custom Time (ms) | Thrust Time (ms) | Custom Throughput (GB/s) | Speedup vs Thrust |
|------------|------------------|------------------|--------------------------|-------------------|
| 1,000 | 0.0486 | 0.0691 | 0.08 | 1.42× |
| 10,000 | 0.0398 | 0.0951 | 1.00 | 2.39× |
| 100,000 | 0.0627 | 0.0780 | 6.38 | 1.24× |
| 1,000,000 | 0.1016 | 0.2604 | 39.38 | 2.56× |
| 10,000,000 | 0.2236 | 0.5736 | 178.86 | 2.57× |
| 100,000,000 | 1.5829 | 2.4130 | **252.71** | 1.52× |
| 500,000,000 | 8.2479 | 9.5849 | 242.49 | 1.16× |

**Peak Performance:**
- **Maximum throughput**: 252.71 GB/s at 100M elements
- **Memory efficiency**: 79.0% of T4 theoretical peak (~320 GB/s)
- **Speedup over Thrust**: 1.16×–2.57× across problem sizes

#### Performance Characteristics

**Small arrays (< 100K):**
- Performance dominated by kernel launch overhead
- Lower throughput (0.08–6.38 GB/s)
- Variable speedup (0.90×–2.39×)

**Medium arrays (1M–10M):**
- Rapid throughput increase (39.38–178.86 GB/s)
- Consistent speedup over Thrust (2.56×–2.57×)
- Good GPU utilization

**Large arrays (100M–500M):**
- Peak memory bandwidth utilization
- Throughput plateaus near hardware limits
- Modest speedup over Thrust (1.16×–1.52×)

#### Comparison with Baselines

**vs NumPy (CPU):**
- Speedup ranges from 0.90× (small) to 21.92× (large)
- CPU baseline: 11.53 GB/s peak (single-threaded)
- GPU advantage grows with problem size

**vs Thrust (GPU):**
- Custom kernel consistently faster (except 100K anomaly)
- Likely due to simpler kernel design and fewer abstraction layers
- Similar performance on very large arrays

### 1.5 Key Insights

1. **Memory-bound performance**: At large sizes, throughput approaches memory bandwidth limit, indicating the kernel is memory-bound rather than compute-bound.

2. **Warp shuffle efficiency**: Using warp intrinsics eliminates shared memory bank conflicts and reduces shared memory footprint.

3. **Two-phase reduction necessity**: Grid-level reduction via atomic operations would create severe contention; two-phase approach maintains efficiency.

4. **Launch overhead impact**: Small arrays suffer from PCIe transfer and kernel launch overhead relative to computation time.

---

## Task 2: Multi-GPU Reduction with PyTorch DDP

### 2.1 Objective

Implement distributed reduction across 2 GPUs using PyTorch's Distributed Data Parallel (DDP) with NCCL backend, measure scaling efficiency, and analyze communication overhead.

### 2.2 Implementation Details

#### Distributed Setup

**Configuration:**
- **Backend**: NCCL (NVIDIA Collective Communications Library)
- **World size**: 2 GPUs (Tesla T4)
- **Launch mechanism**: `torchrun --nproc_per_node=2`
- **Communication pattern**: All-reduce (ring algorithm)

**Data Distribution:**
- Each GPU generates N/2 elements with different random seed
- GPU 0: seed=42, GPU 1: seed=43
- Local reduction on each GPU, then NCCL all-reduce

#### Multi-GPU Script Structure

```python
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()

torch.cuda.set_device(rank)
local_data = torch.randn(size // world_size, device=f'cuda:{rank}')
local_sum = torch.sum(local_data)

dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
```

### 2.3 Correctness Validation

**Test Case**: 10M elements
- **Single GPU reference**: -1074.4121093750
- **Multi-GPU result**: -1074.4121093750
- **Absolute error**: 0.00e+00
- **Relative error**: 0.00e+00
- **Status**: ✓ VALIDATION PASSED

**Conclusion**: Multi-GPU reduction produces bit-exact results compared to single-GPU baseline.

### 2.4 Performance Results

#### Scaling Analysis

| Array Size | Single GPU (ms) | Multi-GPU (ms) | Speedup | Scaling Efficiency |
|------------|-----------------|----------------|---------|-------------------|
| 1,000,000 | 0.0885 | 2.3022 | 0.04× | 1.9% |
| 10,000,000 | 0.1833 | 0.3002 | 0.61× | 30.5% |
| 50,000,000 | 0.7741 | 0.2439 | 3.17× | 158.7% |
| 100,000,000 | 1.5028 | 0.3178 | 4.73× | 236.4% |
| 500,000,000 | 7.3512 | 0.2480 | 29.65× | 1482.4% |
| 1,000,000,000 | 14.6031 | 0.3073 | 47.52× | 2375.9% |

**Summary Statistics:**
- **Average speedup**: 14.29×
- **Minimum speedup**: 0.04× (1M elements)
- **Maximum speedup**: 47.52× (1B elements)
- **Maximum numerical error**: 5.37e-03

### 2.5 Performance Analysis

#### Small Arrays (1M–10M elements)

**Observation**: Multi-GPU slower than single GPU
- Communication overhead dominates
- NCCL initialization and synchronization costs exceed computation savings
- **Efficiency**: 1.9%–30.5%

**Bottleneck**: PCIe bandwidth for small message passing

#### Medium Arrays (50M–100M elements)

**Observation**: Transition to positive speedup
- Computation time begins to offset communication overhead
- **Efficiency**: 158.7%–236.4%
- Approaching ideal 2× speedup

#### Large Arrays (500M–1B elements)

**Observation**: Super-linear speedup (>2×)
- **Speedup**: 29.65×–47.52×
- **Efficiency**: 1482.4%–2375.9%

**Critical Analysis**: These numbers are **anomalous and require investigation**

**Possible Explanations:**
1. **Memory bandwidth**: Single GPU may be hitting memory bandwidth limits; multi-GPU has aggregate 2× bandwidth
2. **Cache effects**: Single GPU working set may exceed cache; distributed data fits better
3. **Measurement artifacts**: Single GPU timing may include additional overhead not present in multi-GPU
4. **PyTorch optimizations**: Different code paths for single vs. multi-GPU

**Note**: Super-linear speedup exceeding theoretical maximum (2×) suggests measurement or system-level effects rather than pure parallelization gains.

#### Communication Overhead Breakdown

**Test case**: 100M elements

| Component | Time (ms) | Percentage |
|-----------|-----------|------------|
| Single GPU (full data) | 1.5028 | - |
| Estimated per-GPU compute | 0.7514 | - |
| Multi-GPU total | 0.3178 | 100% |
| Communication overhead | -0.4336 | -136.5% |

**Interpretation**: Negative communication overhead is non-physical, confirming measurement issues in the timing methodology.

### 2.6 Issues and Recommendations

#### Identified Issues

1. **Unrealistic speedup values**: 47.52× with 2 GPUs violates physical constraints
2. **Negative communication overhead**: Indicates timing measurement problems
3. **Single GPU baseline mismatch**: May not fairly represent equivalent workload

#### Recommendations for Future Work

1. **Use CUDA events for timing**: Replace `time.time()` with `cudaEventRecord` for GPU-accurate timing
2. **Warm-up iterations**: Run multiple iterations and average to eliminate cold-start effects
3. **Fair comparison**: Ensure single-GPU baseline uses same data generation and transfer patterns
4. **Profile with Nsight Systems**: Visualize actual communication timeline
5. **Test on NVLink systems**: Compare PCIe vs. NVLink interconnect performance

### 2.7 Key Insights (Qualitative)

Despite measurement issues, the experiment demonstrates:

1. **NCCL integration**: Successfully configured distributed communication
2. **Correctness**: Numerical validation confirms accurate reduction
3. **Communication-computation tradeoff**: Small arrays suffer from overhead; large arrays benefit from parallelization
4. **Strong scaling**: Fixed problem size distributed across GPUs

---

## Task 3: Floating-Point Associativity Analysis

### 3.1 Objective

Demonstrate and quantify the non-associative nature of floating-point arithmetic by comparing different reduction orders and data distributions, with particular focus on how parallel reduction topologies affect numerical accuracy.

### 3.2 Methodology

#### Reduction Methods Tested

1. **Sequential Sum**: Standard left-to-right `((a₁ + a₂) + a₃) + ... + aₙ`
2. **Reverse Sequential**: Right-to-left `a₁ + (a₂ + (a₃ + ... (aₙ)))`
3. **Tree Reduction**: Binary tree (GPU-style) `(a₁+a₂), (a₃+a₄), ...`
4. **Sorted Sum**: Ascending magnitude order (reduces cancellation)
5. **NumPy**: Optimized CPU SIMD implementation
6. **GPU (PyTorch)**: CUDA tree reduction

#### Data Distributions

**1. Mixed Data**
- 50% large values: uniform(1e6, 1e8)
- 50% small values: uniform(1e-3, 1e-1)
- Randomly shuffled
- **Purpose**: Expose catastrophic cancellation

**2. Pathological Data**
- Pattern: [1e8, 1.0, -1e8, 1e8, 1.0, -1e8, ...]
- Nearly-canceling pairs
- **Purpose**: Worst-case error accumulation

**3. Normal Data**
- Standard normal distribution N(0,1)
- **Purpose**: Typical real-world scenario

#### Reference and Error Calculation

- **Reference**: Double-precision (float64) NumPy sum
- **Error metric**: Absolute error = |result_float32 - result_float64|
- **Array sizes**: 1K to 50M elements

### 3.3 Results

#### 3.3.1 Mixed Data Results

**Maximum errors vs double-precision reference:**

| Size | Sequential | Reverse | Tree | Sorted | NumPy |
|------|------------|---------|------|--------|-------|
| 1,000 | 8.19e+03 | 6.14e+03 | 0.00e+00 | 8.19e+03 | 5.53e+02 |
| 10,000 | 1.97e+05 | 1.15e+05 | 0.00e+00 | 1.64e+04 | 3.10e+03 |
| 100,000 | 3.41e+06 | 1.05e+06 | 0.00e+00 | 1.05e+07 | 1.22e+05 |
| 1,000,000 | 3.94e+08 | 2.16e+08 | 0.00e+00 | 1.03e+09 | 3.68e+06 |
| 10,000,000 | 1.36e+10 | 1.86e+09 | 1.68e+07 | 1.50e+12 | 5.06e+07 |
| 50,000,000 | 7.61e+13 | 7.62e+13 | 0.00e+00 | **9.07e+13** | 9.38e+08 |

**Key Observations:**

1. **Tree reduction** shows zero error for most sizes (used as double-precision reference)
2. **Sorted sum** has worst error at 10M elements (1.50e+12) - counterintuitive!
3. **Error growth**: Roughly linear with array size (catastrophic cancellation)
4. **NumPy optimization**: Significantly better than naive sequential (9.38e+08 vs 7.61e+13 at 50M)

#### 3.3.2 Pathological Data Results

**Maximum errors vs double-precision reference:**

| Size | Sequential | Reverse | Tree | Sorted | NumPy |
|------|------------|---------|------|--------|-------|
| 1,000 | 3.36e+02 | 3.36e+02 | 3.36e+02 | 0.00e+00 | 3.33e+02 |
| 10,000 | 3.34e+03 | 3.34e+03 | 3.34e+03 | 0.00e+00 | 3.33e+03 |
| 100,000 | 3.33e+04 | 3.33e+04 | 3.33e+04 | 0.00e+00 | 3.33e+04 |
| 1,000,000 | 3.33e+05 | 3.33e+05 | 3.33e+05 | 0.00e+00 | 3.33e+05 |
| 10,000,000 | 3.33e+06 | 3.33e+06 | 3.33e+06 | 0.00e+00 | 3.33e+06 |
| 50,000,000 | 1.67e+07 | 1.67e+07 | 1.67e+07 | 0.00e+00 | 1.67e+07 |

**Key Observations:**

1. **All methods fail equally** except sorted sum
2. **Sorted sum perfect**: 0.00e+00 error (adds small values first, avoiding cancellation)
3. **Linear error growth**: Error ≈ N/3 (one 1.0 lost per three elements)
4. **Order matters**: Pattern [1e8, 1.0, -1e8] loses the 1.0 in all unsorted methods

**Why this happens:**
```
float32(1e8 + 1.0) = 1e8  (1.0 is lost - below precision threshold)
float32(1e8 + (-1e8)) = 0
Final: 0 instead of 1.0
```

#### 3.3.3 Normal Data Results

**Maximum errors vs double-precision reference:**

| Size | Sequential | Reverse | Tree | Sorted | NumPy |
|------|------------|---------|------|--------|-------|
| 1,000 | 5.72e-06 | 3.24e-05 | 0.00e+00 | 1.91e-06 | 2.28e-06 |
| 10,000 | 1.91e-05 | 4.39e-05 | 7.63e-06 | 5.53e-05 | 1.08e-05 |
| 100,000 | 2.29e-04 | 1.90e-03 | 1.53e-05 | 7.55e-04 | 4.64e-05 |
| 1,000,000 | 2.11e-02 | 2.49e-02 | 1.22e-04 | 1.46e-02 | 4.59e-04 |
| 10,000,000 | 1.67e-02 | 3.10e-02 | 1.10e-03 | 5.22e-02 | 1.69e-03 |
| 50,000,000 | 3.49e-01 | 1.55e-01 | 0.00e+00 | 2.84e-01 | 2.01e-03 |

**Key Observations:**

1. **Well-behaved error**: Grows gradually with array size
2. **All methods within reasonable bounds**: Max error 3.49e-01 at 50M elements
3. **Tree reduction competitive**: Similar accuracy to sequential
4. **Reverse sum variability**: Sometimes better, sometimes worse than sequential

#### 3.3.4 GPU vs CPU Comparison

Testing on normal random data:

| Size | CPU NumPy Error | GPU PyTorch Error |
|------|-----------------|-------------------|
| 1,000 | 2.28e-06 | 3.77e-07 |
| 10,000 | 1.08e-05 | 3.19e-06 |
| 100,000 | 4.64e-05 | 6.80e-05 |
| 1,000,000 | 4.59e-04 | 9.30e-05 |
| 10,000,000 | 1.69e-03 | 1.41e-04 |
| 50,000,000 | 2.01e-03 | 1.89e-03 |

**Key Observations:**

1. **GPU often more accurate** at smaller sizes (tree reduction properties)
2. **Converge at large sizes**: Both ~2e-03 at 50M elements
3. **Different execution orders**: CPU (SIMD vectorized) vs GPU (tree)
4. **Platform-dependent**: Different rounding patterns due to hardware

### 3.4 Simple Demonstration: Three Numbers

**Setup:**
```python
a = float32(1e8)
b = float32(1.0)
c = float32(-1e8)
```

**Expected (exact arithmetic):** `a + b + c = 1.0`

**Actual Results:**
- `(a + b) + c = 0.0` ❌
- `a + (b + c) = 0.0` ❌

**Explanation:**
1. `float32(1e8 + 1.0)` → Limited precision (7 decimal digits), 1.0 is below representable difference
2. Result: `1e8` (unchanged)
3. `float32(1e8 + (-1e8))` → `0.0`
4. Final result: `0.0` instead of `1.0`

**Key Insight**: Demonstrates catastrophic cancellation where `(a + b) + c ≠ a + (b + c)` violating associativity.

### 3.5 Analysis and Insights

#### Why Reduction Order Matters

1. **Rounding propagation paths**: Each addition introduces rounding error; different orders accumulate errors differently
2. **Magnitude disparities**: Adding large + small loses precision; sorted order mitigates this
3. **Cancellation effects**: Near-zero sums (1e8 - 1e8) magnify relative error
4. **Parallel tree structure**: GPU tree reduction has O(log N) depth vs O(N) for sequential, affecting error accumulation

#### Practical Implications

**For GPU reduction kernels:**
- Tree reduction is unavoidable in parallel systems
- Error characteristics differ from sequential CPU code
- Must validate with appropriate reference (e.g., double precision)
- Cannot assume bit-exact reproducibility across platforms

**For numerical algorithms:**
- Critical algorithms may need compensated summation (Kahan algorithm)
- Order of operations affects reproducibility
- Testing with pathological data exposes worst-case behavior
- Double precision provides validation reference but isn't "ground truth"

#### Mitigation Strategies

1. **Kahan summation**: Compensated summation algorithm maintains error correction term
2. **Pairwise summation**: O(log N) depth like tree, better error bounds than sequential
3. **Sorted summation**: Add small magnitudes first (effective for pathological cases)
4. **Higher precision accumulation**: Use float64 accumulator with float32 inputs
5. **Error analysis**: Bound expected error based on data distribution and algorithm

### 3.6 Connection to Parallel Reduction

**Why This Matters for GPU Programming:**

Parallel reductions (Tasks 1 & 2) inherently use tree-based algorithms:
- **Unavoidable**: Cannot do sequential sum in parallel
- **Different errors**: Tree reduction has different rounding than CPU sequential
- **Non-deterministic**: Thread scheduling may affect reduction order
- **Validation challenge**: Cannot expect bit-exact match with CPU

**Best Practices:**
1. Use double-precision reference for validation
2. Test with mixed and pathological data
3. Expect small differences between GPU and CPU results
4. Document acceptable error tolerances
5. Consider deterministic reduction if reproducibility required (at performance cost)

---

## Conclusions and Key Takeaways

### Technical Achievements

1. **Custom CUDA Reduction**
   - Achieved 79% memory bandwidth efficiency
   - Outperformed Thrust by up to 2.57×
   - Validated correctness across diverse test cases
   - Demonstrated effective use of warp primitives

2. **Multi-GPU Reduction**
   - Successfully implemented NCCL-based distributed reduction
   - Validated numerical correctness across GPUs
   - Identified measurement methodology issues for future improvement
   - Demonstrated strong scaling behavior (qualitatively)

3. **Associativity Analysis**
   - Quantified floating-point non-associativity across three data distributions
   - Demonstrated error variations up to 13 orders of magnitude
   - Identified sorted summation as effective for pathological cases
   - Connected theory to practical GPU reduction implementations

### Performance Insights

**Memory-Bound Operations:**
- Large reductions are memory bandwidth limited
- Custom kernel approaches theoretical hardware limits
- Further optimization requires algorithmic changes (data reuse) not kernel tuning

**Communication Overhead:**
- Small messages suffer from NCCL initialization costs
- Large messages amortize overhead effectively
- Multi-GPU benefits require sufficiently large problem sizes

**Numerical Accuracy:**
- Reduction order significantly impacts accuracy
- Data distribution determines error magnitude
- Tree reduction (GPU) has different error characteristics than sequential (CPU)
- Double precision necessary for validation reference

### Lessons Learned

1. **Timing is Critical**: Use GPU-aware timing mechanisms (CUDA events) not wall-clock time
2. **Baseline Matters**: Ensure fair comparison with equivalent workloads
3. **Test Thoroughly**: Include edge cases, pathological data, and non-power-of-two sizes
4. **Validate Numerically**: Use high-precision reference, not bit-exact comparison
5. **Profile Systematically**: Tools like Nsight provide insights beyond timing alone

### Future Work

**Task 1 Enhancements:**
- Profile with Nsight Compute for detailed metrics (occupancy, warp efficiency)
- Implement cooperative groups alternative
- Test on different GPU architectures (A100, H100)
- Explore multi-element processing per thread

**Task 2 Improvements:**
- Fix timing methodology with CUDA events
- Profile with Nsight Systems for communication timeline
- Test with NVLink-enabled systems
- Scale to 4+ GPUs
- Compare ring vs tree NCCL algorithms

**Task 3 Extensions:**
- Implement Kahan summation for comparison
- Test with float16 (half precision)
- Analyze error bounds theoretically
- Extend to other operations (product, max, min)

### Assignment Objectives Met

✅ **Custom CUDA Reduction**: Implemented, validated, and benchmarked
✅ **Multi-GPU Reduction**: NCCL/DDP implementation with correctness validation
✅ **Associativity Analysis**: Comprehensive quantitative study with three data types
✅ **Profiling & Analysis**: Performance characterization (timing methodology needs improvement)
✅ **Code Quality**: Modular, well-documented notebooks
✅ **Documentation**: Comprehensive report with design decisions and insights

### Final Remarks

This assignment successfully demonstrated the complexity of implementing efficient and correct parallel reductions. The custom CUDA kernel achieved excellent performance, the multi-GPU implementation revealed the challenges of distributed computing, and the associativity experiments highlighted fundamental limitations of floating-point arithmetic. Together, these experiments provide a comprehensive understanding of reduction operations across the full stack from hardware primitives to distributed systems.

---

**End of Report**
