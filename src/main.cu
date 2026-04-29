#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void sgemm_kernel_1(const float* A, const float* B, float* C, int m, int n, int k);
__global__ void sgemm_kernel_2(const float* A, const float* B, float* C, int m, int n, int k);
__global__ void sgemm_kernel_3(const float* A, const float* B, float* C, int m, int n, int k);
__global__ void sgemm_kernel_4(const float* A, const float* B, float* C, int m, int n, int k);
__global__ void sgemm_kernel_5(const float* A, const float* B, float* C, int m, int n, int k);
__global__ void sgemm_kernel_6(const float* A, const float* B, float* C, int m, int n, int k);
__global__ void sgemm_kernel_7(const float* A, const float* B, float* C, int m, int n, int k);

typedef void (*sgemm_kernel_launcher)(
    dim3 grid, dim3 block,
    const float*, const float*, float*,
    int m, int n, int k);

void launch_sgemm_1(dim3 grid, dim3 block,
                    const float* A, const float* B, float* C,
                    int m, int n, int k)
{
    sgemm_kernel_1<<<grid, block>>>(A, B, C, m, n, k);
}

void launch_sgemm_2(dim3 grid, dim3 block,
                    const float* A, const float* B, float* C,
                    int m, int n, int k)
{
    sgemm_kernel_2<<<grid, block>>>(A, B, C, m, n, k);
}

void launch_sgemm_3(dim3 grid, dim3 block,
                    const float* A, const float* B, float* C,
                    int m, int n, int k)
{
    sgemm_kernel_3<<<grid, block>>>(A, B, C, m, n, k);
}

void launch_sgemm_4(dim3 grid, dim3 block,
                    const float* A, const float* B, float* C,
                    int m, int n, int k)
{
    sgemm_kernel_4<<<grid, block>>>(A, B, C, m, n, k);
}
void launch_sgemm_5(dim3 grid, dim3 block,
                    const float* A, const float* B, float* C,
                    int m, int n, int k)
{
    sgemm_kernel_5<<<grid, block>>>(A, B, C, m, n, k);
}

void launch_sgemm_6(dim3 grid, dim3 block,
                    const float* A, const float* B, float* C,
                    int m, int n, int k)
{
    sgemm_kernel_6<<<grid, block>>>(A, B, C, m, n, k);
}

void launch_sgemm_7(dim3 grid, dim3 block,
                    const float* A, const float* B, float* C,
                    int m, int n, int k)
{
    sgemm_kernel_7<<<grid, block>>>(A, B, C, m, n, k);
}

// Utility function to initialize a matrix with random values
void initialize_matrix(float* matrix, int m, int n) 
{
    for (int i = 0; i < m * n; ++i)
        matrix[i] = static_cast<float>((rand() % 101) - 50);
}

// Compare two floating-point vectors
bool compare_matrices(const vector<float>& A, const vector<float>& B, float eps = 1e-5f) 
{
    for (size_t i = 0; i < A.size(); ++i) {
        float abs_diff = fabs(A[i] - B[i]);
        float max_val = max(1.0f, max(fabs(A[i]), fabs(B[i])));

        // Relative error check: if abs_diff is within tolerance relative to the max value
        if (abs_diff > eps * max_val) {
            cout << "Mismatch at index " << i << ": " << A[i] << " != " << B[i] << endl;
            return false;
        }
    }
    return true;
}

float bench_kernel(int kernelID, const float *d_A, const float *d_B, float *d_C, int m, int n, int k)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    sgemm_kernel_launcher kernel_table[7] = {
        launch_sgemm_1,
        launch_sgemm_2,
        launch_sgemm_3,
        launch_sgemm_4,
        launch_sgemm_5,
        launch_sgemm_6,
        launch_sgemm_7
    };

    dim3 thread, grid;

    if (kernelID == 1) {
        thread = dim3(32, 32);
        grid = dim3((n + 31) / 32, (m + 31) / 32);
    } else if (kernelID == 2) {
        thread = dim3(32, 32);
        grid = dim3((m + 31) / 32, (n + 31) / 32);
    } else if (kernelID == 3) {
        thread = dim3(32, 32);
        grid = dim3((m + 31) / 32, (n + 31) / 32);
    } else if (kernelID == 4) {
        thread = dim3(16, 16);
        grid = dim3((m + 127) / 128, (n + 127) / 128);
    } else if (kernelID == 5) {
        thread = dim3(16, 16);
        grid = dim3((m + 127) / 128, (n + 127) / 128);
    } else if (kernelID == 6) {
        thread = dim3(16, 16);
        grid = dim3((m + 127) / 128, (n + 127) / 128);
    } else if (kernelID == 7) {
        thread = dim3(32);
        grid = dim3((n + 7) / 8, (m + 15) / 16);
    }

    // warm-up (push clock to the maximum)
    for (int i = 0; i < 5; ++i)
        kernel_table[kernelID-1](grid, thread, d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();  // Ensure warm-up completes

    // actual measurements
    const int nIters = 10;
    float totalTime = 0.0f;

    for (int i = 0; i < nIters; ++i) {
        cudaEventRecord(start);
        kernel_table[kernelID-1](grid, thread, d_A, d_B, d_C, m, n, k);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop); // Ensure the kernel completes

        float elapsed = 0;
        cudaEventElapsedTime(&elapsed, start, stop);

        totalTime += elapsed / 1000.0f;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float avgTime = totalTime / nIters;
    float FLOPS = 2.0f * m * n * k;

    // Return GFLOPS/s
    return FLOPS / avgTime / 1e9f;

}

float bench_cuBLAS(const float *d_A, const float *d_B, float *d_C, int m, int n, int k)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm-up (push clock to the maximum)
    for (int i = 0; i < 5; ++i)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 
                    &alpha, d_A, m, d_B, k, &beta, d_C, m);
    cudaDeviceSynchronize();  // Ensure warm-up completes

    // actual measurements
    const int nIters = 1;
    float totalTime = 0.0f;

    for (int i = 0; i < nIters; ++i) {
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 
                    &alpha, d_A, m, d_B, k, &beta, d_C, m);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop); // Ensure the kernel completes

        float elapsed = 0;
        cudaEventElapsedTime(&elapsed, start, stop);

        totalTime += elapsed / 1000.0f;
    }

    // Clean up cuBLAS handle
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float avgTime = totalTime / nIters;
    float FLOPS = 2.0f * m * n * k;

    // Return GFLOPS/s
    return FLOPS / avgTime / 1e9f;
}

int main() 
{
    cout << "Welcome to kernelBench" << endl;

    for (int i = 1; i < 9; ++i) {
        // Matrix dimensions
        int m = i * 256, n = i * 256, k = i * 256;

        // Allocate memory for matrices on the host
        vector<float> A(m * k), B(k * n), C(m * n), C_ref(m * n);

        // Randomly initialize matrices A and B
        initialize_matrix(A.data(), m, k);
        initialize_matrix(B.data(), k, n);

        // Allocate memory for matrices on the device (GPU)
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, m * k * sizeof(float));
        cudaMalloc(&d_B, k * n * sizeof(float));
        cudaMalloc(&d_C, m * n * sizeof(float));

        // Copy matrices A and B to device memory
        cudaMemcpy(d_A, A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);

        // A) === cuBLAS kernel ===
        double GFLOPSs = bench_cuBLAS(d_A, d_B, d_C, m, n, k);
        cout << "cuBLAS\t" << m << "\t" << n << "\t" << k << "\t" << GFLOPSs << endl; 

        // store of copy of the cuBLAS result in C_ref to validate correctness
        cudaMemcpy(C_ref.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

        // B) === kernels  ===
        for (int kernelID = 1; kernelID <= 7; ++kernelID) {
            // fill d_C with zeros
            fill(C.begin(), C.end(), 0.0f);
            cudaMemcpy(d_C, C.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

            GFLOPSs = bench_kernel(kernelID, d_A, d_B, d_C, m, n, k);
            cout << "kernel" << kernelID << "\t" << m << "\t" << n << "\t" << k << "\t" << GFLOPSs << endl;

            // validate correctness
            cudaMemcpy(C.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
            compare_matrices(C, C_ref);
        }
       
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
    return EXIT_SUCCESS;
}
