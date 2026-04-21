# Inline PTX Optimization for GELU Activation

**Author:** Gaurav Upadhyay  
**Institution:** Indian Institute of Technology (IIT), Guwahati | M.Tech Computer Science and Engineering  
**Target Role:** Software Intern, PTX Compiler Team (NVIDIA)

## Project Overview
This repository contains a technical demonstration of bridging high-level C++ systems programming with low-level hardware execution. The project implements a custom CUDA kernel for the Gaussian Error Linear Unit (GELU) activation function, utilizing inline PTX (Parallel Thread Execution) assembly to directly override the compiler's instruction selection.

By dropping into the `asm volatile` block and forcing the use of the `fma.rn.f32` (Fused Multiply-Add, round-to-nearest-even) instruction, this project demonstrates an understanding of the PTX Instruction Set Architecture (ISA), the CUDA programming model, and GPU hardware abstraction.

## Performance & Hardware Profiling (Nsight Compute)
To verify the PTX injection and understand the hardware-level execution, the kernels were deeply profiled using NVIDIA Nsight Compute (`ncu`) on a T4 GPU.

### 1. Instruction-Level Verification (The Proof)
The inline PTX successfully forced the compiler to utilize hardware FMA. Nsight Compute instruction statistics confirmed exactly a 1:1 shift of 1,048,576 operations from non-fused FP32 instructions to fused FP32 instructions. This proves the `fma.rn.f32` override was successfully applied across all 1 million threads.

### 2. Bottleneck Analysis (Memory-Bound Workload)
While the instruction footprint was optimized, hardware execution time remained constant at **~42.6 µs** for 1 million elements. Profiling revealed the kernel is strictly **Memory-Bound**, achieving **65.8% DRAM Throughput** but only **32.5% Compute Throughput**. Because the operational intensity (FLOPs per byte) of the GELU activation is relatively low, optimizing the instruction pipeline via PTX successfully reduced compute overhead, but the overall speedup was masked by the broader memory-latency bottleneck.

### 3. Mathematical Fidelity
The PTX-optimized kernel maintained strict mathematical accuracy, yielding a maximum divergence of **2.38419e-07** compared to the CPU ground-truth.

## How to Build and Run
Ensure you have the NVIDIA CUDA Toolkit installed and `nvcc` in your path.

```bash
# Compile the project with maximum optimization
make

# Run the executable to see timing and accuracy checks
./gelu_project

# Run deep hardware profiling (requires Nsight Compute)
ncu --set full ./gelu_project
