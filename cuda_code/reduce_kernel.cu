#include <cuda_runtime.h>
#include <stdio.h>

// Warp-level reduction using shuffle instructions
__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction: shared memory + warp shuffle
__global__ void reduceBlock(float* input, float* output, int n) {
    __shared__ float shared[32];  // One element per warp
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    
    // Grid-stride loop to handle arbitrary input sizes
    float sum = 0.0f;
    for (int i = idx; i < n; i += gridSize) {
        sum += input[i];
    }
    
    // Warp-level reduction
    int warpId = tid / 32;
    int laneId = tid % 32;
    sum = warpReduce(sum);
    
    // First thread in each warp writes to shared memory
    if (laneId == 0) {
        shared[warpId] = sum;
    }
    __syncthreads();
    
    // Final reduction: first warp reduces all warp results
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? shared[tid] : 0.0f;
        sum = warpReduce(sum);
        
        // First thread writes block result
        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

// Two-phase grid-level reduction
extern "C" {
    void launchReduce(float* d_input, float* d_output, int n, 
                      int blockSize, int gridSize) {
        // Phase 1: Reduce within each block
        reduceBlock<<<gridSize, blockSize>>>(d_input, d_output, n);
        cudaDeviceSynchronize();
        
        // Phase 2: Reduce block results (if multiple blocks)
        if (gridSize > 1) {
            reduceBlock<<<1, blockSize>>>(d_output, d_output, gridSize);
            cudaDeviceSynchronize();
        }
    }
}