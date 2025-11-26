#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {
    // Thrust reduce wrapper
    float thrustReduce(float* d_input, int n) {
        thrust::device_ptr<float> ptr(d_input);
        float result = thrust::reduce(ptr, ptr + n, 0.0f, thrust::plus<float>());
        return result;
    }
}