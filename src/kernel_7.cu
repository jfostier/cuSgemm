// kernel_7.cu — TF32 Tensor Core GEMM via mma.sync.m16n8k4
// Column-major matrices (matches cuBLAS + kernels 1–6).

#include <cuda_runtime.h>

__device__ __forceinline__ unsigned float_to_tf32(float x) {
    unsigned y;
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(y) : "f"(x));
    return y;
}

__global__ __launch_bounds__(32)
void sgemm_kernel_7(const float* A, const float* B, float* C,
                    int m, int n, int k)
{
    const int tile_m = blockIdx.y * 16;
    const int tile_n = blockIdx.x * 8;

    const int lane  = threadIdx.x & 31;
    const int gid   = lane >> 2;        // 0..7
    const int lig   = lane &  3;        // 0..3

    // Double-buffered K=4 tiles: sA 16×4, sB 4×8
    __shared__ float sA[2][16 * 4];
    __shared__ float sB[2][4  * 8];

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    int stage = 0, next = 1;

    // ---- preload k0 = 0 ----
    {
        int tid = threadIdx.x;
        // A (col-major): A[i + j*lda], lda = m
        int r = tid % 16, c = tid / 16;  // c = 0 or 1
        sA[stage][r * 4 + c * 2    ] = A[(tile_m + r) + (0 + c * 2    ) * m];
        sA[stage][r * 4 + c * 2 + 1] = A[(tile_m + r) + (0 + c * 2 + 1) * m];
        // B (col-major): B[i + j*ldb], ldb = k
        sB[stage][tid % 4 * 8 + tid / 4] = B[(tid % 4) + (tile_n + tid / 4) * k];
    }
    __syncthreads();

    for (int k0 = 0; k0 < k; k0 += 4) {
        if (k0 + 4 < k) {
            int tid = threadIdx.x;
            int r = tid % 16, c = tid / 16;
            int nk = k0 + 4;
            sA[next][r * 4 + c * 2    ] = A[(tile_m + r) + (nk + c * 2    ) * m];
            sA[next][r * 4 + c * 2 + 1] = A[(tile_m + r) + (nk + c * 2 + 1) * m];
            sB[next][tid % 4 * 8 + tid / 4] = B[(nk + tid % 4) + (tile_n + tid / 4) * k];
            __syncthreads();
        }

        unsigned a0 = float_to_tf32(sA[stage][gid * 4 + lig]);
        unsigned a1 = float_to_tf32(sA[stage][(gid + 8) * 4 + lig]);
        unsigned b0 = float_to_tf32(sB[stage][lig * 8 + gid]);

        float c0 = d0, c1 = d1, c2 = d2, c3 = d3;
        asm volatile(
            "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
            "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(b0),
              "f"(c0), "f"(c1), "f"(c2), "f"(c3));

        stage = next;
        next ^= 1;
    }

    // Store (column-major: C[i + j*lda], lda = m)
    const int row0 = tile_m + gid;
    const int row1 = tile_m + gid + 8;
    const int col0 = tile_n + lig * 2;

    if (row0 < m && col0   < n) C[row0   + col0   * m] = d0;
    if (row0 < m && col0+1 < n) C[row0   + (col0+1) * m] = d1;
    if (row1 < m && col0   < n) C[row1   + col0   * m] = d2;
    if (row1 < m && col0+1 < n) C[row1   + (col0+1) * m] = d3;
}
