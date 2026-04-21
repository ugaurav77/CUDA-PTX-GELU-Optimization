#include <iostream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define M_PI 3.14159265358979323846

// ---------------------------------------------------------
// Day 1: CPU Ground Truth
// ---------------------------------------------------------
void gelu_cpu(const float* x, float* y, int n) {
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    for (int i = 0; i < n; ++i) {
        float val = x[i];
        float cube = val * val * val;
        float inner = sqrt_2_over_pi * (val + 0.044715f * cube);
        y[i] = 0.5f * val * (1.0f + std::tanh(inner));
    }
}

// ---------------------------------------------------------
// Day 2: Naive CUDA Kernel
// ---------------------------------------------------------
__global__ void gelu_naive_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        float cube = val * val * val;
        float inner = 0.79788456f * (val + 0.044715f * cube); 
        y[idx] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

// ---------------------------------------------------------
// Day 4: Inline PTX Device Function (Fused Multiply-Add)
// ---------------------------------------------------------
__device__ inline float ptx_fma(float a, float b, float c) {
    float result;
    // fma.rn.f32: Fused multiply-add, round-to-nearest-even
    asm volatile ("fma.rn.f32 %0, %1, %2, %3;" 
                  : "=f"(result)  
                  : "f"(a), "f"(b), "f"(c)); 
    return result;
}

// ---------------------------------------------------------
// Day 5: PTX-Optimized CUDA Kernel
// ---------------------------------------------------------
__global__ void gelu_ptx_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        float sqr = val * val;
        float cube = val * sqr;
        
        // Use PTX FMA for: poly = (0.044715 * cube) + val
        float poly = ptx_fma(0.044715f, cube, val);
        
        // Use PTX FMA for: inner = (0.79788456 * poly) + 0.0
        float inner = ptx_fma(0.79788456f, poly, 0.0f); 
        
        y[idx] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

// ---------------------------------------------------------
// Day 6: Execution, Benchmarking, and Verification
// ---------------------------------------------------------
int main() {
    int n = 1 << 20; // 1,048,576 elements (1 Million floats)
    size_t bytes = n * sizeof(float);

    // 1. Allocate Host Memory
    std::vector<float> h_x(n);
    std::vector<float> h_y_cpu(n);
    std::vector<float> h_y_naive(n);
    std::vector<float> h_y_ptx(n);

    // Initialize with random numbers between -3.0 and 3.0
    for (int i = 0; i < n; ++i) {
        h_x[i] = ((float)rand() / RAND_MAX) * 6.0f - 3.0f;
    }

    // 2. Allocate Device Memory
    float *d_x, *d_y_naive, *d_y_ptx;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y_naive, bytes);
    cudaMalloc(&d_y_ptx, bytes);

    // 3. Copy Data to Device
    cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice);

    // 4. Set Grid and Block Dimensions
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // 5. Run CPU Baseline
    gelu_cpu(h_x.data(), h_y_cpu.data(), n);

    // Setup CUDA Timing Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 6. Benchmark Naive Kernel
    cudaEventRecord(start);
    gelu_naive_kernel<<<gridSize, blockSize>>>(d_x, d_y_naive, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_ms = 0;
    cudaEventElapsedTime(&naive_ms, start, stop);

    // 7. Benchmark PTX Kernel
    cudaEventRecord(start);
    gelu_ptx_kernel<<<gridSize, blockSize>>>(d_x, d_y_ptx, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ptx_ms = 0;
    cudaEventElapsedTime(&ptx_ms, start, stop);

    // 8. Copy Results Back to Host
    cudaMemcpy(h_y_naive.data(), d_y_naive, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y_ptx.data(), d_y_ptx, bytes, cudaMemcpyDeviceToHost);

    // 9. Verify Accuracy
    float max_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = std::abs(h_y_cpu[i] - h_y_ptx[i]);
        if (diff > max_error) max_error = diff;
    }

    // 10. Print Results
    std::cout << "--- PTX Compiler Project Results ---\n";
    std::cout << "Elements processed: " << n << "\n";
    std::cout << "Naive CUDA Time   : " << naive_ms << " ms\n";
    std::cout << "PTX Optimized Time: " << ptx_ms << " ms\n";
    std::cout << "Max Error (CPU vs PTX): " << max_error << "\n";

    // Clean up
    cudaFree(d_x); cudaFree(d_y_naive); cudaFree(d_y_ptx);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}
